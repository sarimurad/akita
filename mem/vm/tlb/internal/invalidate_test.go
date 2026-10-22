package internal

import (
	"testing"

	"github.com/sarchlab/akita/v3/mem/vm"
)

func fill(s Set, pid vm.PID, vAddr uint64) int {
	w, ok := s.Evict()
	if !ok {
		panic("no way")
	}
	s.Update(w, vm.Page{PID: pid, VAddr: vAddr, Valid: true})
	s.Visit(w)
	return w
}

// An invalidated region is gone and its way is the next one evicted.
func TestRegionSetInvalidate(t *testing.T) {
	s := NewRegionSet(8, 20, 16)
	ways := make([]int, 8)
	for i := 0; i < 8; i++ {
		ways[i] = fill(s, 1, uint64(i)<<20)
	}
	s.(interface{ Invalidate(int) }).Invalidate(ways[5])
	if _, found := s.RegionWay(1, 5<<20); found {
		t.Fatal("region still resident after Invalidate")
	}
	if p := s.PageAt(ways[5]); p.Valid {
		t.Fatal("PageAt reports a valid occupant")
	}
	if w, _ := s.Evict(); w != ways[5] {
		t.Fatalf("Evict returned way %d, want invalidated way %d", w, ways[5])
	}
	for i := 0; i < 8; i++ {
		if i == 5 {
			continue
		}
		if _, found := s.RegionWay(1, uint64(i)<<20); !found {
			t.Fatalf("region %d lost", i)
		}
	}
}

// Same for the page-granular set.
func TestSetInvalidate(t *testing.T) {
	s := NewSet(8)
	ways := make([]int, 8)
	for i := 0; i < 8; i++ {
		ways[i] = fill(s, 1, uint64(i)<<16)
	}
	s.(interface{ Invalidate(int) }).Invalidate(ways[2])
	if _, _, found := s.Lookup(1, 2<<16); found {
		t.Fatal("page still found after Invalidate")
	}
	if w, _ := s.Evict(); w != ways[2] {
		t.Fatalf("Evict returned way %d, want %d", w, ways[2])
	}
}
