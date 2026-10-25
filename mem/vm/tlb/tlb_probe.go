// This file adds a non-intrusive probe interface used to model a Prime+Probe
// attacker against the shared last-level TLB. It is appended to package tlb
// (same package as tlb.go) as tlb_probe.go.
//
// The method reads residency directly rather than inferring it from response
// latency. On real hardware the attacker distinguishes a resident translation
// from an evicted one by timing its own re-access: a resident entry returns
// quickly, an evicted one only after a page walk. That latency difference
// exists precisely because the residency state differs, so reading residency
// measures the same channel at its source. Crucially the probe does not call
// Visit, so measuring does not perturb the LRU state being measured.

package tlb

import "github.com/sarchlab/akita/v3/mem/vm"

// EvictionSet is a group of virtual addresses, all owned by one process, that
// the attacker has arranged to map to the same TLB set. Under the region model
// each address should fall in a distinct region so that, together, they occupy
// every way of the target set.
type EvictionSet struct {
	PID    vm.PID
	VAddrs []uint64
}

// SetIDOf exposes the TLB's set-index mapping so an attacker component can
// construct eviction sets against the same hash the victim's accesses use.
func (tlb *TLB) SetIDOf(vAddr uint64) int {
	return tlb.vAddrToSetID(vAddr)
}

// NumSets reports the number of sets, so a probe can sweep all of them.
func (tlb *TLB) NumSets() int {
	return tlb.numSets
}

// NumWays reports the associativity, i.e. the maximum contention level.
func (tlb *TLB) NumWays() int {
	return tlb.numWays
}

// ProbeResident reports, for one eviction set, how many of its addresses are
// still resident in the TLB. This is the contention measurement for the set
// those addresses map to: a value of k means the victim (or another tenant)
// displaced len(VAddrs) - k of the attacker's primed entries since they were
// installed.
//
// The lookup is read-only. It uses the region-residency query where available
// so that it neither inserts, evicts, nor reorders anything, and therefore
// does not itself disturb the state it is measuring.
func (tlb *TLB) ProbeResident(es EvictionSet) (resident int) {
	for _, vAddr := range es.VAddrs {
		setID := tlb.setIDFor(es.PID, vAddr)
		set := tlb.Sets[setID]

		if rs, ok := set.(interface {
			RegionWay(pid vm.PID, vAddr uint64) (int, bool)
		}); ok {
			if _, found := rs.RegionWay(es.PID, vAddr); found {
				resident++
			}
			continue
		}

		// Fall back to a plain lookup for the page-granular set. Lookup does
		// not call Visit, so it is safe to use here as a pure query.
		if _, _, found := set.Lookup(es.PID, vAddr); found {
			resident++
		}
	}

	return resident
}

// Contention is the complement of ProbeResident: how many of the attacker's
// primed entries were evicted. This is the quantity a real probe recovers, a
// value between 0 and len(es.VAddrs).
func (tlb *TLB) Contention(es EvictionSet) (evicted int) {
	return len(es.VAddrs) - tlb.ProbeResident(es)
}

// Prime installs every address of an eviction set into the TLB, so that a
// later probe can detect which of them the victim has since displaced. It
// drives the ordinary fill path so that priming is subject to the same
// replacement policy as any other access.
//
// now is the current simulated time. The caller supplies it because priming
// generates fills that the replacement policy times like any other.
func (tlb *TLB) Prime(es EvictionSet) {
	for _, vAddr := range es.VAddrs {
		setID := tlb.setIDFor(es.PID, vAddr)
		set := tlb.Sets[setID]

		// Skip if already resident, matching a real prime that only needs to
		// (re)touch entries that are not currently present.
		if tlb.isResident(set, es.PID, vAddr) {
			continue
		}

		wayID, ok := set.Evict()
		if !ok {
			continue
		}
		set.Update(wayID, vm.Page{
			PID:   es.PID,
			VAddr: vAddr,
			Valid: true,
		})
		set.Visit(wayID)
	}
}

func (tlb *TLB) isResident(set interface{}, pid vm.PID, vAddr uint64) bool {
	if rs, ok := set.(interface {
		RegionWay(pid vm.PID, vAddr uint64) (int, bool)
	}); ok {
		_, found := rs.RegionWay(pid, vAddr)
		return found
	}
	if s, ok := set.(interface {
		Lookup(pid vm.PID, vAddr uint64) (int, vm.Page, bool)
	}); ok {
		_, _, found := s.Lookup(pid, vAddr)
		return found
	}
	return false
}

// BuildEvictionSet returns an eviction set of `ways` addresses that all map to
// targetSet, spaced by the region size so each occupies a distinct region. It
// searches the address space linearly for addresses whose set index matches,
// which is inexpensive because the GF(2) mapping distributes 1 MB-spaced
// addresses across sets in a regular pattern.
//
// regionStride should be the region size (e.g. 1<<20 for 1 MB regions) so that
// consecutive members land in different regions rather than sharing one.
func (tlb *TLB) BuildEvictionSet(
	pid vm.PID,
	targetSet int,
	ways int,
	regionStride uint64,
) EvictionSet {
	es := EvictionSet{PID: pid}
	var vAddr uint64

	for len(es.VAddrs) < ways {
		if tlb.setIDFor(pid, vAddr) == targetSet {
			es.VAddrs = append(es.VAddrs, vAddr)
		}
		vAddr += regionStride
		if vAddr == 0 { // wrapped without filling: give up
			break
		}
	}

	return es
}

// BuildAllEvictionSets builds one eviction set per TLB set for pid, under
// pid's current mapping, in a single sweep over region-spaced addresses. A
// keyed mapping scatters candidates over sets at random, so building all sets
// at once is far cheaper than calling BuildEvictionSet 256 times. It panics if
// maxCandidates addresses are not enough to fill every set.
func (tlb *TLB) BuildAllEvictionSets(
	pid vm.PID,
	ways int,
	regionStride uint64,
	maxCandidates int,
) []EvictionSet {
	sets := make([]EvictionSet, tlb.numSets)
	for s := range sets {
		sets[s].PID = pid
	}

	remaining := tlb.numSets
	vAddr := regionStride // skip address 0
	for i := 0; remaining > 0 && i < maxCandidates; i++ {
		s := tlb.setIDFor(pid, vAddr)
		if len(sets[s].VAddrs) < ways {
			sets[s].VAddrs = append(sets[s].VAddrs, vAddr)
			if len(sets[s].VAddrs) == ways {
				remaining--
			}
		}
		vAddr += regionStride
	}

	if remaining > 0 {
		panic("BuildAllEvictionSets: candidate budget too small")
	}
	return sets
}
