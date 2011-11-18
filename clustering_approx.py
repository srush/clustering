from clustering import *
from heapq import *

INF = 1e6
class Edge:
  def __cmp__(self, other):
    return cmp(self.cost, other.cost)
  
  def __init__(self, city, facility, cost):
    self.city = city
    self.facility = facility
    self.cost = cost

class CityInfo:
  def __init__(self, city):
    self.city = city
    self.connected = None
    self.tight_facilities = []
    self.special_facilities = []
    self.is_connected = False

  def MakeConnected(self, facility, time):
    assert(not self.is_connected)
    self.is_connected = True

    # Remove this city from all tight facilities. 
    for facility_info in self.tight_facilities:
      if facility_info.facility.id <> facility.id:
        facility_info.RemoveCity(self, time)
      
  def AddTightEdge(self, facility_info, time):
    self.tight_facilities.append(facility_info)
    facility_info.AddWitnessCity(self)
    if facility_info.temporarily_open:
      self.MakeConnected(facility_info.facility, time)
      return 1
    else:
      facility_info.AddCity(self, time)
      self.special_facilities.append(facility_info)
      return 0
    
class FacilityInfo:
  def __init__(self, facility, cost):
    self.facility = facility
    self.time_to_pay = INF
    self.paying_cities = set()
    self.witness_cities = []
    self.special_cities = []
    self.temporarily_open = False
    self.cost_left = cost
    self.last_update_time = None
    
  def __cmp__(self, other):
    return cmp(self.time_to_pay, other.time_to_pay)

  def UpdateTimeToPay(self, time):
    if not self.paying_cities:
      self.time_to_pay = INF
    else:
      self.time_to_pay = time + (self.cost_left / len(self.paying_cities))

  def UpdateCost(self, time):
    if self.last_update_time <> None:
      self.cost_left -= (time - self.last_update_time) * len(self.paying_cities)
    self.last_update_time = time
    
  def TemporarilyOpen(self, time):
    assert(not self.temporarily_open)
    self.temporarily_open = True

    # All cities paying are now connected. 
    for city in self.paying_cities:
      city.MakeConnected(self.facility, time)
    return len(self.paying_cities)
  
  def AddCity(self, city, time):
    self.UpdateCost(time)
    self.paying_cities.add(city)

    # Sometimes paying, always special.
    self.special_cities.append(city)
    self.UpdateTimeToPay(time)

  def AddWitnessCity(self, city):
    self.witness_cities.append(city)
    
  def RemoveCity(self, city, time):
    self.UpdateCost(time)
    self.paying_cities.remove(city)
    self.UpdateTimeToPay(time)

class ClusteringApprox:
  def __init__(self):
    pass
  
  def build(self, clustering_problem):
    cp = clustering_problem

    # Create variables
    # Connections
    x = {}
    # Facilities
    y = {}
    edges = []
    facility_pay_heap = []
    facility_infos = {}
    city_infos = {} 
    for f in cp.facilities:
      info = FacilityInfo(f, cp.ScoreOpenFacility(f))
      facility_pay_heap.append(info)
      facility_infos[f.id] = info
      for c in cp.cities:
        score = cp.ScoreCityAssignment(c, f)
        edges.append(Edge(c, f, score))

    for c in cp.cities:
      info = CityInfo(c)
      city_infos[c.id] = info
      
    # Sort edges in increasing order
    print "sort start", len(edges)
    
    edges.sort()
    print "sort done"
    time = 0
    edge_position = 0 
    connected_count = 0
    
    # End loop when all cities are connected.
    while connected_count < len(cp.cities):
      time_to_edge = edges[edge_position].cost
      time_to_pay = 0
      if facility_pay_heap:
        time_to_pay = facility_pay_heap[0].time_to_pay
      else :
        time_to_pay = INF
        #print time_to_edge, time_to_pay
      if time_to_edge < time_to_pay:
        city = edges[edge_position].city
        if city_infos[city.id].is_connected:
          # City is already connected, skip. 
          edge_position += 1
        else: 
          # Edge went tight, update.
          time = time_to_edge
          #print "City", time

          # Start paying for the facility. 
          facility = edges[edge_position].facility
          facility_info = facility_infos[facility.id]
          connected_count += city_infos[city.id].AddTightEdge(facility_info, time)
          edge_position += 1
      else:
        time = time_to_pay
        #print "Facility", time
        # Facility is completely paid for, temporarily open.
        next_open = heappop(facility_pay_heap)
        connected_count += next_open.TemporarilyOpen(time)

      heapify(facility_pay_heap)
#     for f in facility_infos.itervalues():
#       print f.facility.id, f.temporarily_open
#     for c in city_infos.itervalues():
#       print c.city.id, " ".join([str(f.facility.id) for f in c.tight_facilities])

    # Phase 2
    # Construct facility graph
    graph = []
    graph_order = facility_infos.values()
    graph_order.sort(lambda x, y: cmp(len(y.special_cities),
                                      len(x.special_cities)))
    for f in graph_order:
      neighbors = []
      for c in f.special_cities:
        for f2 in c.special_facilities:
          neighbors.append(f2.facility.id)
      graph.append((f.facility, neighbors))

    chosen_facilities, killer_neighbor = FindMaximalIndependentSet(graph)
    chosen_facilities_ids = set([f.id for f in chosen_facilities])
#     for f in chosen_facilities:
#       print f.id

    assignment = {}
    for c in city_infos.itervalues():
      score = INF
      for f in chosen_facilities:
        cur_score = cp.ScoreCityAssignment(c.city, f)
        if cur_score < score:
          assignment[c.city] = f
          score = cur_score
#       # First try special edges.
#       assigned = False
#       for f in c.special_facilities:
#         if f.facility.id in chosen_facilities_ids:
#           assignment[c.city] = f.facility
#           assigned = True
#           break
        
#       # Next try tight edges.
#       if not assigned:
#         score = INF
#         for f in c.tight_facilities:
#           if f.facility.id in chosen_facilities_ids:
#             assigned = True
#             cur_score = cp.ScoreCityAssignment(c.city, f.facility)
#             if cur_score < score:
#               assignment[c.city] = f.facility
#               score = cur_score
#       # Finally try all one-hop edges.
#       if not assigned:
#         score = INF
#         for f in c.tight_facilities:
#           for f2 in killer_neighbor[f.facility.id]:
#             cur_score = cp.ScoreCityAssignment(c.city, f2)
#             if cur_score < score:
#               assignment[c.city] = f2
#               score = cur_score
              
    return ClusteringAnswer(cp, assignment) 
def FindMaximalIndependentSet(adjacency):
  killed = {}
  s = []
  for node, neighbors in adjacency:
    if node.id not in killed:
      s.append(node)
      for n in neighbors:
        killed.setdefault(n, [])
        killed[n].append(node)
  return s, killed
