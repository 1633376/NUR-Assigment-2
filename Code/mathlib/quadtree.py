import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections

class QuadTree(object):

    def __init__(self, left_bottom, size, max_boddies):
        self._root = Quad(left_bottom, size, max_boddies, None)
   
    def add_boddy(self, boddy):
        if boddy[0]  < 30: # and boddy[1] > 120:
            print(boddy[0], boddy[1])

        self._root.add_boddy(boddy)
       
    def find_leaf(self, pos_x, pos_y):
        
        leaf = self._root

        while not leaf._leaf:
            leaf = leaf._find_quad(pos_x, pos_y)
        return leaf

    def print_moments(self, pos_x, pos_y):
        leaf = self.find_leaf(pos_x, pos_y)

        print(leaf._moment)
        parent = leaf._parent

        while parent._parent != None: # not root
            print(parent._moment)
            parent = parent._parent
        # root
        print(parent._moment)

    def plot(self):

        axis = plt.gca()
        rectangles = list()

        self._root._plot(axis, rectangles)
        axis.add_collection(collections.PatchCollection(rectangles, match_original=True))

        plt.xlim(self._root._left_bottom[0], self._root._left_bottom[1] + self._root._size)
        plt.ylim(self._root._left_bottom[1], self._root._left_bottom[1] + self._root._size)
        plt.show()

class Quad(object):

    def __init__(self, left_bottom, size, max_boddies, parent = None):

        # Geometric properties
        self._left_bottom = left_bottom
        self._size = size
        self._halve_size = size/2

        # 'Facts' at the current moment of time
        self._max_boddies = max_boddies
        self._leaf = True
        self._parent = parent

        # sub quads and children
        self._child_quads = None
        self._boddies = list()

        # moment
        self._moment = 0


    def add_boddy(self, boddy):
        
        
        # if current quad is a leaf, add the item or split
        if not self._leaf:
            # Not a leaf or splitted, find the quad that 
            # should hold the item and add it to that quad
            self._find_quad(boddy[0], boddy[1]).add_boddy(boddy)    
            return

        # Can still add the item
        if len(self._boddies) < self._max_boddies:
            self._boddies.append(boddy)
            self._update_moment(boddy[3])
        else: # split
            # on split moment is automatically updates
            self._boddies.append(boddy) # add it and split, (given to children)
            self._split()
        
      
        
    def _find_quad(self, pos_x, pos_y):
             
        
        if self._leaf:
            return self

        # right top or right bottom
        if pos_x > self._halve_size+self._left_bottom[0]:
            if pos_y >= self._halve_size+self._left_bottom[1]:
                return self._child_quads[2]  # right top
            else:
                return self._child_quads[3]  # right bottom

        elif pos_y >= self._halve_size+self._left_bottom[1]:
            return self._child_quads[0] # left top
        else:   
            return self._child_quads[1] # left bottom
       
    def _split(self):

        self._child_quads = list()
        
        # left bottom coordinates of child quads
        lt = [self._left_bottom[0], self._left_bottom[1] + self._halve_size]
        lb = self._left_bottom
        rt = [self._left_bottom[0] + self._halve_size, self._left_bottom[1] + self._halve_size]
        rb = [self._left_bottom[0] + self._halve_size, self._left_bottom[1]]

        # add the quads
        self._child_quads.append(Quad(lt, self._halve_size,self._max_boddies,self))
        self._child_quads.append(Quad(lb, self._halve_size,self._max_boddies,self))
        self._child_quads.append(Quad(rt, self._halve_size,self._max_boddies,self))
        self._child_quads.append(Quad(rb, self._halve_size,self._max_boddies,self))

        # Not a leaf anymore
        self._leaf = False

        # Add the boddies of the current quad to the new child quads to the new quads
        for boddy in self._boddies:
            self.add_boddy(boddy) 
        
        # Empty items that where hold by this node
        self._boddies = None

    def _update_moment(self, moment = 0):
        if self._leaf:
            self._moment += moment
        else:
            self._moment = 0
            
            for quad in self._child_quads:
                self._moment += quad._moment 

        if self._parent != None:
            self._parent._update_moment()


    def _plot(self,axis, rectangles):

        rect = patches.Rectangle(self._left_bottom,self._size, self._size, fill=False)
        rectangles.append(rect)

        if self._leaf:
            if len(self._boddies) == 0:
                return
            a = np.array(self._boddies)
            axis.scatter(a[:,0],a[:,1],c='blue',s=1)
        else:
            for quad in self._child_quads:
                quad._plot(axis, rectangles)



                 

    


    


        
        


