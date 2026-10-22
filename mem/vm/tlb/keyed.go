package tlb

// Keyed set-index mapping for the prevention pillar (live key rotation).
//
// The reference platform computes the last-level TLB set index as a public
// linear map over GF(2): output bit r is the XOR of a fixed subset of the
// virtual-address bits VA[46:20]. Each row of that map is stored here as a
// 64-bit mask over the address, and the index is the parity of (VA & row).
//
// The keyed mapping replaces the public matrix with a secret, full-rank
// matrix drawn from a key. This changes WHICH addresses collide, not only
// which label a set carries. (The earlier keyed variant, H(VA ^ key<<20),
// equals H(VA) ^ H(key<<20) because H is linear, so it only relabelled every
// set by a constant XOR: eviction sets stayed congruent across a re-key.)
//
// The construction is still linear, so an attacker that gathers enough
// collisions can solve for it algebraically (report section 5.8.5). That
// weakness is deliberate here: it is the "secret keyed GF(2) matrix"
// candidate, the cheap one that the FPGA prototype will cost.

import (
	"math/bits"

	"github.com/sarchlab/akita/v3/mem/vm"
)

const (
	hashLowBit  = 20
	hashHighBit = 46
	hashOutBits = 8
)

// setMap is one set-index function: hashOutBits row masks over VA[46:20].
type setMap struct {
	rows [hashOutBits]uint64
}

// publishedMap is the reference platform's mapping: output bit r is the XOR
// of every eighth address bit starting at bit 20+r, up to bit 46.
func publishedMap() *setMap {
	m := &setMap{}
	for r := 0; r < hashOutBits; r++ {
		for b := hashLowBit + r; b <= hashHighBit; b += 8 {
			m.rows[r] |= 1 << uint(b)
		}
	}
	return m
}

// index returns the set index of vAddr under this map.
func (m *setMap) index(vAddr uint64) int {
	idx := 0
	for r := 0; r < hashOutBits; r++ {
		idx |= (bits.OnesCount64(vAddr&m.rows[r]) & 1) << uint(r)
	}
	return idx
}

// splitmix64 is a small deterministic generator, so a key always yields the
// same matrix and every run is reproducible.
func splitmix64(state *uint64) uint64 {
	*state += 0x9E3779B97F4A7C15
	z := *state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// keyedMap draws a secret full-rank matrix from key. Key 0 is the published
// mapping, so un-keyed behaviour and the existing hash tests are unchanged.
func keyedMap(key uint64) *setMap {
	if key == 0 {
		return publishedMap()
	}

	field := uint64((1<<(hashHighBit+1))-1) &^ uint64((1<<hashLowBit)-1)
	state := key
	for {
		m := &setMap{}
		for r := 0; r < hashOutBits; r++ {
			m.rows[r] = splitmix64(&state) & field
		}
		if gf2Rank(m.rows[:]) == hashOutBits {
			return m
		}
	}
}

// gf2Rank is the rank over GF(2) of the given row vectors.
func gf2Rank(rows []uint64) int {
	work := append([]uint64(nil), rows...)
	rank := 0
	for bit := 63; bit >= 0; bit-- {
		pivot := -1
		for i := rank; i < len(work); i++ {
			if work[i]>>uint(bit)&1 == 1 {
				pivot = i
				break
			}
		}
		if pivot < 0 {
			continue
		}
		work[rank], work[pivot] = work[pivot], work[rank]
		for i := range work {
			if i != rank && work[i]>>uint(bit)&1 == 1 {
				work[i] ^= work[rank]
			}
		}
		rank++
	}
	return rank
}

// mapFor returns the mapping that applies to pid: its own if one was set with
// RekeyPID, otherwise the TLB-wide default (published unless Rekey was called).
func (tlb *TLB) mapFor(pid vm.PID) *setMap {
	if m, ok := tlb.pidMaps[pid]; ok {
		return m
	}
	if tlb.defaultMap == nil {
		tlb.defaultMap = publishedMap()
	}
	return tlb.defaultMap
}

// setIDFor is the per-tenant set index used on every lookup, fill and flush.
func (tlb *TLB) setIDFor(pid vm.PID, vAddr uint64) int {
	return tlb.mapFor(pid).index(vAddr) % tlb.numSets
}

// Rekey installs a new default mapping drawn from key, for every tenant that
// has no mapping of its own. Key 0 restores the published mapping. Resident
// entries are NOT moved: call FlushExcept/FlushPID afterwards, as the
// hypervisor would, or they are simply never found again.
func (tlb *TLB) Rekey(key uint64) {
	tlb.defaultMap = keyedMap(key)
}

// RekeyPID gives one tenant its own mapping drawn from key.
func (tlb *TLB) RekeyPID(pid vm.PID, key uint64) {
	if tlb.pidMaps == nil {
		tlb.pidMaps = make(map[vm.PID]*setMap)
	}
	tlb.pidMaps[pid] = keyedMap(key)
}

type wayInvalidator interface {
	Invalidate(wayID int)
}

// flushMatching invalidates every valid way whose owner satisfies match and
// returns how many were dropped. Under the region model one way is one 1 MB
// region, so the count is in entries, not pages.
func (tlb *TLB) flushMatching(match func(vm.PID) bool) int {
	n := 0
	for _, set := range tlb.Sets {
		inv, ok := set.(wayInvalidator)
		if !ok {
			panic("TLB set does not support Invalidate")
		}
		for w := 0; w < tlb.numWays; w++ {
			p := set.PageAt(w)
			if p.Valid && match(p.PID) {
				inv.Invalidate(w)
				n++
			}
		}
	}
	tlb.FlushedEntries += uint64(n)
	return n
}

// FlushPID drops every entry owned by pid.
func (tlb *TLB) FlushPID(pid vm.PID) int {
	return tlb.flushMatching(func(p vm.PID) bool { return p == pid })
}

// FlushExcept drops every entry not owned by keep. Used after a re-key of all
// tenants but the one whose mapping did not change.
func (tlb *TLB) FlushExcept(keep vm.PID) int {
	return tlb.flushMatching(func(p vm.PID) bool { return p != keep })
}

// countWalk records one translation sent to the lower level (a page walk from
// this TLB's point of view), per tenant.
func (tlb *TLB) countWalk(pid vm.PID) {
	tlb.Walks++
	if tlb.WalksByPID == nil {
		tlb.WalksByPID = make(map[vm.PID]uint64)
	}
	tlb.WalksByPID[pid]++
}

// SetIDOfPID exposes the per-tenant mapping so a harness can locate where a
// given tenant's address lands under its current key.
func (tlb *TLB) SetIDOfPID(pid vm.PID, vAddr uint64) int {
	return tlb.setIDFor(pid, vAddr)
}
