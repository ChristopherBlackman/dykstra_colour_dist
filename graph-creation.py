# Author : Christopher Blackman
# Citation : GNU General Public License v3
# Description : 
# Implementations of 2 types of shortest path algorithms over image colour distance


from pqdict import minpq
from PIL import Image
import gc
import time
import heapq
import math
import os
import sys
import csv
import random


# graph implementation credited to : http://www.bogotobogo.com/python/python_graph_data_structures.php
# Start 1
class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def __lt__(self,other):
        return 0

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()
# End 1

def pos_to_id(x,y):
    return (x,y)

# generate a graph
def array_2d_to_graph(array,size,modifier):
    G = Graph()
    w,h = size
    print(size)

    print('adding vertices to graph')
    # add all Vertices in graph
    for x in range(w): 
        for y in range(h):
            name = pos_to_id(x,y)
            G.add_vertex(name)           


    print('adding edges to graph')
    # add edges to nodes
    for x in range(w): 
        for y in range(h):
            name = pos_to_id(x,y)
            V = G.get_vertex(name)
            
            
            # connecting edges
            up =  G.get_vertex(pos_to_id(x,y-1))
            down =    G.get_vertex(pos_to_id(x,y+1))
            right = G.get_vertex(pos_to_id(x+1,y))
            left =  G.get_vertex(pos_to_id(x-1,y))

            # addition of edges 
            if not None == up:
                G.add_edge(V.get_id(),up.get_id(),weight(array[x,y],array[x,y-1],modifier))
            if not None == down:
                G.add_edge(V.get_id(),down.get_id(),weight(array[x,y],array[x,y+1],modifier))
            if not None == right:
                G.add_edge(V.get_id(),right.get_id(),weight(array[x,y],array[x+1,y],modifier))
            if not None == left:
                G.add_edge(V.get_id(),left.get_id(),weight(array[x,y],array[x-1,y],modifier))

    print('graph has been created')
    return G

# weight function
def weight(Color_A,Color_B,modifier):
    return 1 + modifier*color_dist(Color_A,Color_B)

# colour distance function
def color_dist(Color_A,Color_B):
    dist = 0
    for i in range(min(len(Color_A),len(Color_B))):
       dist += (Color_A[i]-Color_B[i])**2 
    return math.sqrt(dist)


# decreasing key implementation
def djkstra(G,sources):
    maxh = 0


    print('djkstra starting')
    q = minpq()
    #dist = {source:0}
    dist = init_dist_dict(sources)
    prev = dict()

    temp = set(sources)
    #initialize the priority queue
    for v in G:
        if not v in temp:
            dist[v] = float('inf')
            prev[v] = None
        q[v] = dist[v]
    del temp

    
    for u, min_dist in q.popitems():
        maxh = max(len(q),maxh)
        for n in u.get_connections():
            alt = dist[u] + u.get_weight(n)

            if alt < dist[n]:
                dist[n] = alt
                prev[n] = u
                q[n] = alt
    
    print('djkstra done')
    return (dist,prev,maxh/len(G.get_vertices()))


# heap based priority implementation
def modified_djkstra(G,sources):
    maxh = 0


    print('modified_djkstra starting')
    q = []
    #dist = {source:0}
    dist = init_dist_dict(sources)
    prev = dict()
    
    temp = set(sources)
    for v in G:
        if not v in temp:
            dist[v] = float('inf')
            prev[v] = None
        heapq.heappush(q,(dist[v],v))
    del temp

    while q: 
        #separate function
        maxh = max(len(q),maxh)

        item = heapq.heappop(q)
        p = item[0]
        u = item[1]

        if p > dist[u]:
            pass

        for n in u.get_connections():
            alt = dist[u] + u.get_weight(n)
            if alt < dist[n]:
                dist[n] = alt
                prev[n] = u
                heapq.heappush(q,(dist[n],n))

    print('modified_djkstra done')
    return (dist,prev,maxh/len(G.get_vertices()))

def init_dist_dict(list_sources):
    dist = {}
    for source in list_sources:
        dist[source] = 0
    return dist

def create_image(size):
    width, height = size
    im = Image.new('L', size)
    pix = im.load()
    for x in range(width):
        for y in range(height):
            pix[x,y] = (255)
    return im

def create_image_RGB_consistent(size):
    width, height = size
    im = Image.new('RGB', size)
    pix = im.load()
    for x in range(width):
        for y in range(height):
            pix[x,y] = (0,0,0)
    return im

def create_image_RGB_random(size):
    width, height = size
    im = Image.new('RGB', size)
    pix = im.load()
    for x in range(width):
        for y in range(height):
            pix[x,y] = (int(random.random()*256),int(random.random()*256),int(random.random()*256))
    return im




# create an image from the distance mapping
def find_maximal(dist):
    maxima = (0,0)
    maxima_d = 0
    for v,d in dist.items():
        maxima = max(maxima,v.get_id())
        if maxima == v.get_id():
            maxima_d = d
    return (maxima,maxima_d)

def find_diameter_of_image(dist,prev):
    maxima,d = find_maximal(dist)
    return d

def dist_to_pic(dist):
    print('creating image')
    mapping = dict()
    maxima = (0,0)
    maxima_d = 0
    for v,d in dist.items():
        mapping[v.get_id()] = d
        maxima = max(maxima,v.get_id())
        maxima_d = max(maxima_d,d)

    w,h = maxima  
    array = []
    for x in range(w):
        temp = []
        for y in range(h):
            temp.append(math.floor((mapping[pos_to_id(x,y)]/maxima_d)*255))
        array.append(temp)

    img = create_image(maxima)
    img_p = img.load()

    for x in range(w):
        for y in range(h):
            img_p[x,y] = (array[x][y])
    return img

# general tester functions that tests against implementations
def tester(iterations=3,random=False,output_file_1="decreasing_key.csv",output_file_2='heap_based.csv',sources={'top-left':True,'middle':False},weight=1.0):
    start_time = 0 
    data_d = []
    data_dm = []
    header = ['NoP','time','heap_usage','diameter']


    for n in range(1,iterations): 
        m = n*100
        img = None
        print("n: ",n)

        
        if random:
            img = create_image_RGB_random((m,m))
        else:
            img = create_image_RGB_consistent((m,m))

        G = array_2d_to_graph(img.load(),img.size,weight)

        starting_locations = []
        for source in sources:
            if sources.get('top-left',False):
                starting_locations.append(G.get_vertex(pos_to_id(0,0)))
            if sources.get('middle',False):
                starting_locations.append(G.get_vertex(pos_to_id(int(m/2),int(m/2))))

        
        gc.collect()
        start_time = time.time()
        dist,prev,maxh = djkstra(G,starting_locations)
        data_d.append({'NoP':m*m,'time':(time.time()-start_time),'heap_usage':maxh,'diameter':find_diameter_of_image(dist,prev)})

        gc.collect()
        start_time = time.time()
        dist,prev,maxh = modified_djkstra(G,starting_locations)
        data_dm.append({'NoP':m*m,'time':(time.time()-start_time),'heap_usage':maxh,'diameter':find_diameter_of_image(dist,prev)})

    print('output:',output_file_1)
    
    with open(output_file_1,'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerow({'NoP':output_file_1})

        for item in data_d:
            writer.writerow(item)          
 
    print('output:',output_file_2)

    with open(output_file_2,'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerow({'NoP':output_file_2})

        for item in data_dm:
            writer.writerow(item)          

# add more tests here if need be
def test_suite(n_iter):
    tester(iterations=n_iter,random=False,output_file_1="decreasing_key_consistent_top-left_.csv",output_file_2='heap_based_consistent_top-left_.csv',sources={'top-left':True},weight=1.0)
    tester(iterations=n_iter,random=True,output_file_1="decreasing_key_random_top-left_.csv",output_file_2='heap_based_random_top-left_.csv',sources={'top-left':True},weight=1.0)
    tester(iterations=n_iter,random=False,output_file_1="decreasing_key_consistent_middle_.csv",output_file_2='heap_based_consistent_middle_.csv',sources={'middle':True},weight=1.0)
    tester(iterations=n_iter,random=True,output_file_1="decreasing_key_random_middle_.csv",output_file_2='heap_based_random_middle_.csv',sources={'middle':True},weight=1.0)
    tester(iterations=n_iter,random=False,output_file_1="decreasing_key_consistent_middle_top-left_.csv",output_file_2='heap_based_consistent_middle_top-left_.csv',sources={'top-left':True,'middle':True},weight=1.0)
    tester(iterations=n_iter,random=True,output_file_1="decreasing_key_random_middle_top-left_.csv",output_file_2='heap_based_random_middle_top-left_.csv',sources={'top-left':True,'middle':True},weight=1.0)

# computes the image on both algorithms
def compute_image(input_path,output_path,modifier=1.0):
    img = None
    img_d = None
    img_p = None
    start_time = 0 

    try:
        img = Image.open(input_path).convert('RGB')
    except IOError:
        print(IOError)
        return 

    G = array_2d_to_graph(img.load(),img.size,modifier)
    h,w = img.size

    nodes = [G.get_vertex(pos_to_id(0,0))]
    #nodes = [G.get_vertex(pos_to_id(0,0)),G.get_vertex(pos_to_id(int(h/2),int(w/2)))]
    
    start_time = time.time()
    dist,prev,heap = djkstra(G,nodes)
    print('time taken : ', time.time()-start_time)
    print('heap usage :{0}'.format(heap))

    img_d = dist_to_pic(dist)
    img_d.save(output_path+"_decreasing_key.png","png")
     
    start_time = time.time()
    dist,prev,heap = modified_djkstra(G,nodes)
    print('time taken : ', time.time()-start_time)
    print('heap usage :{0}'.format(heap))

    img_p = dist_to_pic(dist)
    img_p.save(output_path+"_heap_based_priority.png","png")
   

# main function
def main():
    # python3 -f path/input.png path/output.png
    if len(sys.argv) == 5:
        if  sys.argv[1] == "-f" and os.path.exists(sys.argv[2] and sys.argv[4].isnumeric()):
            inputfile  = sys.argv[2]
            outputfile = sys.argv[3]
            modifier = float(sys.argv[4])
            compute_image(inputfile,outputfile,modifier)
    elif len(sys.argv) == 3:
        if sys.argv[1] == "-t" and sys.argv[2].isnumeric():
            iterations = int(sys.argv[2])
            test_suite(iterations)
    else:
        print("Options")
        print("-f 'PathToInputFile' 'PathToOutputFile' Modifier(Number)")
        print("-t Iterations(Number)")


if __name__ == '__main__':
    main()

     
