package tlb

import (
	"math/rand"
	"testing"

	"github.com/sarchlab/akita/v3/mem/vm"
)

// Tests for the keyed set mapping, re-keying and flushing used by the live
// key-rotation experiment. Plain testing (no Ginkgo) so they run alone.

func newRegionTLB() *TLB {
	t := &TLB{numSets: 256, numWays: 8, pageSize: 1 << 16, log2RegionSize: 20}
	t.reset()
	return t
}

func randVA(rng *rand.Rand) uint64 {
	mask := uint64((1<<47)-1) &^ uint64((1<<20)-1)
	return rng.Uint64() & mask
}

// Key 0 must be exactly the published mapping, so un-keyed runs reproduce
// Chapter 7 bit for bit.
func TestKeyZeroIsPublishedMapping(t *testing.T) {
	tl := &TLB{numSets: 256, pageSize: 1 << 16}
	m := keyedMap(0)
	rng := rand.New(rand.NewSource(1234))
	for i := 0; i < 10000; i++ {
		v := randVA(rng)
		if m.index(v) != tl.vAddrToSetID(v) {
			t.Fatalf("key 0 differs from published at %#x", v)
		}
	}
}

// Every key must give a full-rank (surjective) map, reproducibly.
func TestKeyedMapFullRankAndDeterministic(t *testing.T) {
	for k := uint64(1); k <= 64; k++ {
		m := keyedMap(k)
		if gf2Rank(m.rows[:]) != 8 {
			t.Fatalf("key %d: rank < 8", k)
		}
		if *m != *keyedMap(k) {
			t.Fatalf("key %d: not deterministic", k)
		}
		if *m == *publishedMap() {
			t.Fatalf("key %d: equals the published map", k)
		}
	}
}

// The point of re-keying: an eviction set built under one key must stop being
// an eviction set under the next. Count, over all 256 sets, how many stay
// congruent (all 8 members still in one set). Chance is about 256^-7.
func TestRekeyBreaksEvictionSetCongruence(t *testing.T) {
	tl := newRegionTLB()
	sets := tl.BuildAllEvictionSets(vm.PID(9), 8, 1<<20, 1<<22)
	for k := uint64(1); k <= 16; k++ {
		m := keyedMap(k)
		congruent := 0
		for _, es := range sets {
			first := m.index(es.VAddrs[0])
			same := true
			for _, v := range es.VAddrs[1:] {
				if m.index(v) != first {
					same = false
					break
				}
			}
			if same {
				congruent++
			}
		}
		if congruent > 0 {
			t.Fatalf("key %d: %d of 256 eviction sets still congruent", k, congruent)
		}
	}
}

// Documents why the previous keyed variant was replaced: H(VA ^ key<<20) is
// H(VA) ^ H(key<<20), so every eviction set stays congruent after a re-key.
func TestOldXorKeyOnlyRelabelsSets(t *testing.T) {
	tl := newRegionTLB()
	sets := tl.BuildAllEvictionSets(vm.PID(9), 8, 1<<20, 1<<22)
	pub := publishedMap()
	for k := uint64(1); k <= 16; k++ {
		for _, es := range sets {
			first := pub.index(es.VAddrs[0] ^ (k << 20))
			for _, v := range es.VAddrs[1:] {
				if pub.index(v^(k<<20)) != first {
					t.Fatalf("old XOR key %d broke congruence (unexpected)", k)
				}
			}
		}
	}
}

// RekeyPID changes only that tenant's mapping.
func TestRekeyPIDIsPerTenant(t *testing.T) {
	tl := newRegionTLB()
	tl.RekeyPID(vm.PID(1), 7)
	m7, pub := keyedMap(7), publishedMap()
	rng := rand.New(rand.NewSource(5))
	for i := 0; i < 1000; i++ {
		v := randVA(rng)
		if tl.SetIDOfPID(1, v) != m7.index(v) {
			t.Fatal("PID 1 not using its key")
		}
		if tl.SetIDOfPID(2, v) != pub.index(v) {
			t.Fatal("PID 2 should keep the published map")
		}
	}
}

// BuildAllEvictionSets under a keyed map yields 8 distinct regions per set,
// all congruent under that map.
func TestBuildAllEvictionSetsKeyed(t *testing.T) {
	tl := newRegionTLB()
	tl.RekeyPID(vm.PID(9), 3)
	sets := tl.BuildAllEvictionSets(vm.PID(9), 8, 1<<20, 1<<22)
	for s, es := range sets {
		if len(es.VAddrs) != 8 {
			t.Fatalf("set %d has %d members", s, len(es.VAddrs))
		}
		for _, v := range es.VAddrs {
			if tl.SetIDOfPID(9, v) != s {
				t.Fatalf("member %#x of set %d maps elsewhere", v, s)
			}
		}
	}
}

// Prime fills every way; FlushPID drops exactly that tenant's entries and
// leaves the other tenant's intact.
func TestFlushPIDDropsOnlyThatTenant(t *testing.T) {
	tl := newRegionTLB()
	a := tl.BuildAllEvictionSets(vm.PID(1), 4, 1<<20, 1<<22)
	b := tl.BuildAllEvictionSets(vm.PID(2), 4, 1<<20, 1<<22)
	for s := range a {
		tl.Prime(a[s])
		tl.Prime(b[s])
	}
	for s := range a {
		if tl.ProbeResident(a[s]) != 4 || tl.ProbeResident(b[s]) != 4 {
			t.Fatalf("set %d not fully primed", s)
		}
	}
	if n := tl.FlushPID(1); n != 256*4 {
		t.Fatalf("FlushPID dropped %d entries, want %d", n, 256*4)
	}
	for s := range a {
		if tl.ProbeResident(a[s]) != 0 {
			t.Fatalf("set %d: PID 1 still resident", s)
		}
		if tl.ProbeResident(b[s]) != 4 {
			t.Fatalf("set %d: PID 2 lost entries", s)
		}
	}
	if tl.FlushedEntries != 256*4 {
		t.Fatalf("FlushedEntries = %d", tl.FlushedEntries)
	}
	// Flushed ways are reused first: priming PID 3 must not evict PID 2.
	c := tl.BuildAllEvictionSets(vm.PID(3), 4, 1<<20, 1<<22)
	for s := range c {
		tl.Prime(c[s])
	}
	for s := range b {
		if tl.ProbeResident(b[s]) != 4 {
			t.Fatalf("set %d: refilling flushed ways evicted PID 2", s)
		}
	}
}

// FlushExcept keeps only the named tenant.
func TestFlushExcept(t *testing.T) {
	tl := newRegionTLB()
	a := tl.BuildAllEvictionSets(vm.PID(1), 4, 1<<20, 1<<22)
	b := tl.BuildAllEvictionSets(vm.PID(2), 4, 1<<20, 1<<22)
	for s := range a {
		tl.Prime(a[s])
		tl.Prime(b[s])
	}
	if n := tl.FlushExcept(vm.PID(1)); n != 256*4 {
		t.Fatalf("FlushExcept dropped %d", n)
	}
	for s := range a {
		if tl.ProbeResident(a[s]) != 4 || tl.ProbeResident(b[s]) != 0 {
			t.Fatalf("set %d wrong after FlushExcept", s)
		}
	}
}
