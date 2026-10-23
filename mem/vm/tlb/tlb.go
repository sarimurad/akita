package tlb

import (
	"log"
	"reflect"
	"math/bits"

	"github.com/sarchlab/akita/v3/mem/vm"
	"github.com/sarchlab/akita/v3/mem/vm/tlb/internal"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
)

// A TLB is a cache that maintains some page information.
type TLB struct {
	*sim.TickingComponent

	topPort     sim.Port
	bottomPort  sim.Port
	controlPort sim.Port

	LowModule sim.Port

	numSets        int
	numWays        int
	pageSize       uint64
	numReqPerCycle int

	Sets []internal.Set

	mshr                mshr
	respondingMSHREntry *mshrEntry

	isPaused bool

	// Instrumentation for the shared-TLB side-channel study.
	EvictValid   uint64
	EvictInvalid uint64
	EvictCrossPID uint64
	log2RegionSize uint64
	SubEntryFill   uint64

	// Keyed set mapping (keyed.go). defaultMap applies to every tenant
	// without its own entry in pidMaps; nil means the published mapping.
	defaultMap *setMap
	pidMaps    map[vm.PID]*setMap

	// Cost accounting for re-keying: entries dropped by flushes, and
	// translations sent to the lower level (page walks), total and per PID.
	FlushedEntries uint64
	Walks          uint64
	WalksByPID     map[vm.PID]uint64
	// FillsByPIDSet records successful translation fills by tenant and set.
	// It supports hypervisor-side footprint measurements without a probe.
	FillsByPIDSet map[vm.PID][]uint64

	// RoundHook, when set, is called at the start of every tick. The live
	// re-keying harness uses it to run probe rounds on simulated time while
	// the TLB is active, without keeping the engine alive on its own.
	RoundHook func(now sim.VTimeInSec)
}

// Reset sets all the entries int he TLB to be invalid
func (tlb *TLB) reset() {
	tlb.Sets = make([]internal.Set, tlb.numSets)
	for i := 0; i < tlb.numSets; i++ {
		if tlb.log2RegionSize > 0 {
			log2Page := uint64(bits.TrailingZeros64(tlb.pageSize))
			tlb.Sets[i] = internal.NewRegionSet(tlb.numWays, tlb.log2RegionSize, log2Page)
		} else {
			tlb.Sets[i] = internal.NewSet(tlb.numWays)
		}
	}
}

// Tick defines how TLB update states at each cycle
func (tlb *TLB) Tick(now sim.VTimeInSec) bool {
	madeProgress := false

	if tlb.RoundHook != nil {
		tlb.RoundHook(now)
	}

	madeProgress = tlb.performCtrlReq(now) || madeProgress

	if !tlb.isPaused {
		for i := 0; i < tlb.numReqPerCycle; i++ {
			madeProgress = tlb.respondMSHREntry(now) || madeProgress
		}

		for i := 0; i < tlb.numReqPerCycle; i++ {
			madeProgress = tlb.lookup(now) || madeProgress
		}

		for i := 0; i < tlb.numReqPerCycle; i++ {
			madeProgress = tlb.parseBottom(now) || madeProgress
		}
	}

	return madeProgress
}

func (tlb *TLB) respondMSHREntry(now sim.VTimeInSec) bool {
	if tlb.respondingMSHREntry == nil {
		return false
	}

	mshrEntry := tlb.respondingMSHREntry
	page := mshrEntry.page
	req := mshrEntry.Requests[0]
	rspToTop := vm.TranslationRspBuilder{}.
		WithSendTime(now).
		WithSrc(tlb.topPort).
		WithDst(req.Src).
		WithRspTo(req.ID).
		WithPage(page).
		Build()
	err := tlb.topPort.Send(rspToTop)
	if err != nil {
		return false
	}

	mshrEntry.Requests = mshrEntry.Requests[1:]
	if len(mshrEntry.Requests) == 0 {
		tlb.respondingMSHREntry = nil
	}

	tracing.TraceReqComplete(req, tlb)
	return true
}

func (tlb *TLB) lookup(now sim.VTimeInSec) bool {
	msg := tlb.topPort.Peek()
	if msg == nil {
		return false
	}

	req := msg.(*vm.TranslationReq)

	mshrEntry := tlb.mshr.Query(req.PID, req.VAddr)
	if mshrEntry != nil {
		return tlb.processTLBMSHRHit(now, mshrEntry, req)
	}

	setID := tlb.setIDFor(req.PID, req.VAddr)
	set := tlb.Sets[setID]
	wayID, page, found := set.Lookup(req.PID, req.VAddr)
	if found && page.Valid {
		return tlb.handleTranslationHit(now, req, setID, wayID, page)
	}

	return tlb.handleTranslationMiss(now, req)
}

func (tlb *TLB) handleTranslationHit(
	now sim.VTimeInSec,
	req *vm.TranslationReq,
	setID, wayID int,
	page vm.Page,
) bool {
	ok := tlb.sendRspToTop(now, req, page)
	if !ok {
		return false
	}

	tlb.visit(setID, wayID)
	tlb.topPort.Retrieve(now)

	tracing.TraceReqReceive(req, tlb)
	tracing.AddTaskStep(tracing.MsgIDAtReceiver(req, tlb), tlb, "hit")
	tracing.TraceReqComplete(req, tlb)

	return true
}

func (tlb *TLB) handleTranslationMiss(
	now sim.VTimeInSec,
	req *vm.TranslationReq,
) bool {
	if tlb.mshr.IsFull() {
		return false
	}

	fetched := tlb.fetchBottom(now, req)
	if fetched {
		tlb.topPort.Retrieve(now)
		tracing.TraceReqReceive(req, tlb)
		tracing.AddTaskStep(tracing.MsgIDAtReceiver(req, tlb), tlb, "miss")
		return true
	}

	return false
}


// vAddrToSetID maps a virtual address to a set index under the TLB-wide
// default mapping: the published GF(2) map over VA[46:20] (output bit r is
// the XOR of every eighth bit from 20+r) unless Rekey installed a keyed one.
// Per-tenant lookups go through setIDFor in keyed.go.
func (tlb *TLB) vAddrToSetID(vAddr uint64) (setID int) {
	if tlb.defaultMap == nil {
		tlb.defaultMap = publishedMap()
	}
	return tlb.defaultMap.index(vAddr) % tlb.numSets
}

func (tlb *TLB) sendRspToTop(
	now sim.VTimeInSec,
	req *vm.TranslationReq,
	page vm.Page,
) bool {
	rsp := vm.TranslationRspBuilder{}.
		WithSendTime(now).
		WithSrc(tlb.topPort).
		WithDst(req.Src).
		WithRspTo(req.ID).
		WithPage(page).
		Build()

	err := tlb.topPort.Send(rsp)

	return err == nil
}

func (tlb *TLB) processTLBMSHRHit(
	now sim.VTimeInSec,
	mshrEntry *mshrEntry,
	req *vm.TranslationReq,
) bool {
	mshrEntry.Requests = append(mshrEntry.Requests, req)

	tlb.topPort.Retrieve(now)
	tracing.TraceReqReceive(req, tlb)
	tracing.AddTaskStep(tracing.MsgIDAtReceiver(req, tlb), tlb, "mshr-hit")

	return true
}

func (tlb *TLB) fetchBottom(now sim.VTimeInSec, req *vm.TranslationReq) bool {
	fetchBottom := vm.TranslationReqBuilder{}.
		WithSendTime(now).
		WithSrc(tlb.bottomPort).
		WithDst(tlb.LowModule).
		WithPID(req.PID).
		WithVAddr(req.VAddr).
		WithDeviceID(req.DeviceID).
		Build()
	err := tlb.bottomPort.Send(fetchBottom)
	if err != nil {
		return false
	}

	tlb.countWalk(req.PID)

	mshrEntry := tlb.mshr.Add(req.PID, req.VAddr)
	mshrEntry.Requests = append(mshrEntry.Requests, req)
	mshrEntry.reqToBottom = fetchBottom

	tracing.TraceReqInitiate(fetchBottom, tlb,
		tracing.MsgIDAtReceiver(req, tlb))

	return true
}

func (tlb *TLB) parseBottom(now sim.VTimeInSec) bool {
	if tlb.respondingMSHREntry != nil {
		return false
	}

	item := tlb.bottomPort.Peek()
	if item == nil {
		return false
	}

	rsp := item.(*vm.TranslationRsp)
	page := rsp.Page

	mshrEntryPresent := tlb.mshr.IsEntryPresent(rsp.Page.PID, rsp.Page.VAddr)
	if !mshrEntryPresent {
		tlb.bottomPort.Retrieve(now)
		return true
	}

	setID := tlb.setIDFor(page.PID, page.VAddr)
	set := tlb.Sets[setID]
	if tlb.FillsByPIDSet == nil {
		tlb.FillsByPIDSet = make(map[vm.PID][]uint64)
	}
	if tlb.FillsByPIDSet[page.PID] == nil {
		tlb.FillsByPIDSet[page.PID] = make([]uint64, tlb.numSets)
	}
	tlb.FillsByPIDSet[page.PID][setID]++

	wayID, resident := -1, false
	if rs, ok := set.(internal.RegionSet); ok {
		wayID, resident = rs.RegionWay(page.PID, page.VAddr)
	}

	if resident {
		// The region already holds a way: fill the sub-entry in place.
		tlb.SubEntryFill++
	} else {
		var ok bool
		wayID, ok = set.Evict()
		if !ok {
			panic("failed to evict")
		}
		victim := set.PageAt(wayID)
		if victim.Valid {
			tlb.EvictValid++
			if victim.PID != page.PID {
				tlb.EvictCrossPID++
			}
		} else {
			tlb.EvictInvalid++
		}
	}
	set.Update(wayID, page)
	set.Visit(wayID)

	mshrEntry := tlb.mshr.GetEntry(rsp.Page.PID, rsp.Page.VAddr)
	tlb.respondingMSHREntry = mshrEntry
	mshrEntry.page = page

	tlb.mshr.Remove(rsp.Page.PID, rsp.Page.VAddr)
	tlb.bottomPort.Retrieve(now)
	tracing.TraceReqFinalize(mshrEntry.reqToBottom, tlb)

	return true
}

func (tlb *TLB) performCtrlReq(now sim.VTimeInSec) bool {
	item := tlb.controlPort.Peek()
	if item == nil {
		return false
	}

	item = tlb.controlPort.Retrieve(now)

	switch req := item.(type) {
	case *FlushReq:
		return tlb.handleTLBFlush(now, req)
	case *RestartReq:
		return tlb.handleTLBRestart(now, req)
	default:
		log.Panicf("cannot process request %s", reflect.TypeOf(req))
	}

	return true
}

func (tlb *TLB) visit(setID, wayID int) {
	set := tlb.Sets[setID]
	set.Visit(wayID)
}

func (tlb *TLB) handleTLBFlush(now sim.VTimeInSec, req *FlushReq) bool {
	rsp := FlushRspBuilder{}.
		WithSrc(tlb.controlPort).
		WithDst(req.Src).
		WithSendTime(now).
		Build()

	err := tlb.controlPort.Send(rsp)
	if err != nil {
		return false
	}

	for _, vAddr := range req.VAddr {
		setID := tlb.setIDFor(req.PID, vAddr)
		set := tlb.Sets[setID]
		wayID, page, found := set.Lookup(req.PID, vAddr)
		if !found {
			continue
		}

		page.Valid = false
		set.Update(wayID, page)
	}

	tlb.mshr.Reset()
	tlb.isPaused = true
	return true
}

func (tlb *TLB) handleTLBRestart(now sim.VTimeInSec, req *RestartReq) bool {
	rsp := RestartRspBuilder{}.
		WithSendTime(now).
		WithSrc(tlb.controlPort).
		WithDst(req.Src).
		Build()

	err := tlb.controlPort.Send(rsp)
	if err != nil {
		return false
	}

	tlb.isPaused = false

	for tlb.topPort.Retrieve(now) != nil {
		tlb.topPort.Retrieve(now)
	}

	for tlb.bottomPort.Retrieve(now) != nil {
		tlb.bottomPort.Retrieve(now)
	}

	return true
}
