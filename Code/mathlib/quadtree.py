import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class QuadTree(object):

    def __init__(self, left_bottom, size, max_boddies):
        self._root = Quad(left_bottom, size, max_boddies,None)

    
    def add_boddy(self, boddy):
        self._root.add_boddy(boddy)
       
    def plot(self):
        self._root._plot()

        plt.xlim(self._root._left_bottom[0], self._root._left_bottom[1] + self._root._size+1)
        plt.ylim(self._root._left_bottom[1], self._root._left_bottom[1] + self._root._size+1)
        plt.show()
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


    def add_boddy(self, boddy):
        
        # if current quad is a leaf, add the item or split
        if self._leaf:

            # Can still add the item
            if len(self._boddies) < self._max_boddies:
                self._boddies.append(boddy)
                return
            # Split
            else:
                self._split()

        # Not a leaf or splitted, find the quad that 
        # should hold the item and add it to that quad
        self._find_quad(boddy[0], boddy[1]).add_boddy(boddy)
        
    def _find_quad(self, pos_x, pos_y):
        

        quad_idx = -1

        # right top or right bottom
        if pos_x > self._halve_size+self._left_bottom[0]:
            # right top
            if pos_y > self._halve_size+self._left_bottom[1]:
                quad_idx = 2
            else:
                # right bottom
                quad_idx = 3

        # left top or left bottom
        else:
            # left top
            if pos_y > self._halve_size+self._left_bottom[1]:
                quad_idx = 0
            # left bottom
            else:
                quad_idx = 1
       

        return self._child_quads[quad_idx]

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

    def _plot(self):

        rect = plt.Rectangle(self._left_bottom,self._size, self._size, fill=False)
        plt.gca().add_patch(rect)

        if self._leaf:
            for boddy in self._boddies:
                plt.scatter(boddy[0],boddy[1],c='blue',s=1)
        else:
            for quad in self._child_quads:
                quad._plot()



                 

    


    


        
        


