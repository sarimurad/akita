package internal

// Invalidate support, used when the hypervisor re-keys the set mapping and
// must drop translations that would otherwise sit, unreachable, under their
// old set index. An invalidated way becomes the least recently used one, so
// it is the next to be refilled, exactly like a way that was never used.

// Invalidate drops the page held in wayID.
func (s *setImpl) Invalidate(wayID int) {
	b := s.blocks[wayID]
	delete(s.vAddrWayIDMap, s.keyString(b.page.PID, b.page.VAddr))
	b.page.Valid = false
	moveToLRU(&s.visitList, b.wayID, func(x *block) int { return x.wayID },
		func() { b.lastVisit = 0 }, b)
}

// Invalidate drops the whole region held in wayID.
func (s *regionSetImpl) Invalidate(wayID int) {
	b := s.blocks[wayID]
	if b.valid {
		delete(s.regionMap, b.key)
	}
	b.valid = false
	b.key = regionKey{}
	for i := range b.subs {
		b.subs[i].Valid = false
	}
	moveToLRU(&s.visitList, b.wayID, func(x *regionBlock) int { return x.wayID },
		func() { b.lastVisit = 0 }, b)
}

// moveToLRU moves item to the front of an LRU-ordered visit list.
func moveToLRU[T any](list *[]T, wayID int, idOf func(T) int, reset func(), item T) {
	l := *list
	for i, x := range l {
		if idOf(x) == wayID {
			l = append(l[:i], l[i+1:]...)
			break
		}
	}
	reset()
	l = append(l, item)
	copy(l[1:], l[:len(l)-1])
	l[0] = item
	*list = l
}
