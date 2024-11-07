def parse(file_path):
    t = Trace()
    for line in open(file_path):
        try:
            timestamp, length = line.strip().split('\t')
            direction = int(int(float(length))) // abs(int(float(length)))
            t.append(Packet(float(timestamp), direction, abs(int(float(length)))))
        except ValueError:
            logger.warn("Could not split line: %s in %s", line, file_path, ValueError)
            continue
    return t


def dump(trace, file_path):
    with open(file_path, 'w') as fo:
        for packet in trace:
            fo.write("{:.6f}".format(packet.timestamp) +'\t' + "{}".format(packet.direction*packet.length)\
                + '\n')


class Packet(object):
    payload = None

    def __init__(self, timestamp, direction, length, dummy=False):
        self.timestamp = timestamp
        self.direction = direction
        self.length = length
        self.dummy = dummy

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __str__(self):
        return '\t'.join(map(str, [self.timestamp, self.direction * self.length]))


class Trace(list):
    _index = 0

    def __init__(self,  list_packets=None):
        if list_packets:
            for p in list_packets:
                self.append(p)

    def __getslice__(self, i, j):
        return Trace(list_packets=list.__getslice__(self, i, j))

    def __add__(self, new_packet):
        t = Trace(self.pcap)
        l = list.__add__(self, new_packet)
        for e in l:
            t.append(e)
        return t

    def __mul__(self, other):
        return Trace(list.__mul__(self, other))

    def get_next_by_direction(self, i, direction):
        # print(i,direction, len(self))
        flag = 0
        for j, p in enumerate(self[i + 1:]):
            if p.direction == direction:
                flag = 1
                return i + j + 1
        if flag == 0:
            return -1

    def next(self):
        try:
            i = self[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return i

class Flow(Trace):
    """Provides a structure to keep flow dependent variables."""

    def __init__(self, direction):
        """Initialize direction and state of the flow."""
        self.direction = direction
        self.expired = False
        self.timeout = 0.0
        self.state = ct.BURST
        Trace.__init__(self)