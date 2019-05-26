import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections

class QuadTree(object):
    """
        An object representing a quad tree
    """

    def __init__(self, left_bottom, size, max_boddies):
        """
            Create an instance of the quadtree.
        In:
            param: left_bottom -- The coordinates of the leftbottom point of 
                                  the root quad.
            param: size -- The size of the root quad.
            param: max_boddies -- The maximum amount of boddies to add before
                                  splitting a node.
        """

        # Create the root quad.
        self._root = Quad(left_bottom, size, max_boddies, None)
   
    def add_boddy(self, boddy):
        """
            Add a boddy to the tree.
        In:
            param: boddy -- The boddy to add.
        """

        # Add the boddy to the root node
        self._root.add_boddy(boddy)
       
    def find_leaf(self, pos_x, pos_y):
        """
            Find the leaf node that contains the specific positon.
        In:
            param: pos_x -- The x coordinate of the position.
            param: pos_y -- The y coordinate of the position.
        Out:
            return: The leaf node that contains the specific positon.
        """

        # Start by assuming that the root node is the node.
        leaf = self._root

        # If it has children find the child containing the position.
        while not leaf._leaf:
            leaf = leaf._find_quad(pos_x, pos_y)
        
        # return the quad that contains the current position.
        return leaf

    def print_moments(self, pos_x, pos_y):
        """
            Print the multipole moment for the leaf node
            and al its parents that contain the given positon.
        In:
            param: pos_x -- The x coordinate of the position.
            param: pos_y -- The y coordinate of the position.
        """

        # Find the leaf that contains the given position.
        leaf = self.find_leaf(pos_x, pos_y)

        # Print the mulitpole moment of the leaf
        print(leaf._moment)

        # Iterate through its parents and print the 
        # multipole moment of the parents
        parent = leaf._parent

        # Only root doesn't have a parent, thus abort
        # when parent is root.
        while parent._parent != None: 
            print(parent._moment)
            parent = parent._parent

        # Don't forget printing the moment of the root node.
        print(parent._moment)

    def plot(self):
        """
            Create a visual representation of the quad tree 
            and the boddies added to the tree.
        """

        # The axis used for plotting
        axis = plt.gca()
        # An list that contains rectangle objects (matplotlib.patches.Rectangle)
        # for each of the quads in the tree. 
        rectangles = list()

        # Fill the list with rectangles by recursively calling the 
        # children of the root. If a quad is a leaf it will furtheremore
        # axis.scatter. This is to add the boddies of that leaf
        # to the plot. The reason that a quad not directly adds its own
        # rectangle to the axis (axis.add_patches) is to save time.
        self._root._plot(axis, rectangles)

        # Add all the rectangles at once to save time.
        axis.add_collection(collections.PatchCollection(rectangles, 
                            match_original=True))

        # Create and save the plot
        plt.xlim(self._root._left_bottom[0], 
                 self._root._left_bottom[1] + self._root._size)
        plt.ylim(self._root._left_bottom[1], 
                 self._root._left_bottom[1] + self._root._size)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('./Plots/7_quadtree.pdf')
        plt.figure()

class Quad(object):

    def __init__(self, left_bottom, size, max_boddies, parent = None):
        """
            Create an instance of a quad in a quadtree.
        In:
            param: left_bottom -- The coordinates of the leftbottom point of 
                                  the  quad.
            param: size -- The size of the  quad.
            param: max_boddies -- The maximum amount of boddies to add before
                                  splitting this quad.
            param: parent -- The parent quad if any.
        """

        # Geometric properties of the current quad
        self._left_bottom = left_bottom
        self._size = size
        self._halve_size = size/2

        # 'Facts' about the quad at the moment of initalization
        self._max_boddies = max_boddies
        self._leaf = True
        self._parent = parent

        # An array containing child quads (if any) and 
        # the boddies in this quad (if it is a leaf).

        # The child quads are named: 
        # Left Top (index 0), Left Bottom (index 1)
        # Right top (index 2), Right bottom (index 3)
        self._child_quads = None
        self._boddies = list()

        # The multipole moment of the current quad.
        self._moment = 0


    def add_boddy(self, boddy):
        """
            Add a boddy to the current quad or to one of its 
            child quads.
        In:
            param: boddy -- The boddy to add. Should be an array
                            in which the first for elemnts respectivelty
                            correspond with the x-position, y-position,
                            z-positon and mass.
        """
        
        
        # If current quad is not a leaf find 
        # the child quad that should hold the boddy.
        if not self._leaf:
            self._find_quad(boddy[0], boddy[1]).add_boddy(boddy)    
            return

        # Current quad is a leaf and can still add a boddy.
        if len(self._boddies) < self._max_boddies:
            # Add the boddy.
            self._boddies.append(boddy)  
            # Update the multipole moment for this quad.
            self._update_moment(boddy[3]) #index 3 = mass.

        # Current quad is a leaf, but must split when adding a new boddy.
        else: 
            # The boddy is before splitting first added
            # to the current quad. When it splits the boddies
            # of this quad will be assigned to the child quads created
            # in the split.
            self._boddies.append(boddy) 
            # Split the current quad and make it a leaf.
            self._split()
        
      
        
    def _find_quad(self, pos_x, pos_y):
        """
            Find a child quad that contains the specific
            position.
        In:
            param: pos_x -- The x coordinate of the position.
            param: pos_y -- The y coordinates of the position.
        Out:
            return: The child quad containing the position. If this 
                     quad is a leaf, then the quad its self is returned.
        """
             
        # Current quad is a leaf, return its self.
        if self._leaf:
            return self

        # Positon is in the right top or right bottom quad.
        if pos_x > self._halve_size+self._left_bottom[0]:
            # Position is in the right top quad.
            if pos_y >= self._halve_size + self._left_bottom[1]:
                return self._child_quads[2]  
            # Position is in the right bottom quad.
            else:
                return self._child_quads[3]  
        # Position is in the left top quad.
        elif pos_y >= self._halve_size + self._left_bottom[1]:
            return self._child_quads[0]
        # Position is in the left bottom quad.
        else:   
            return self._child_quads[1] 
       
    def _split(self):
        """
            Split the current quad in 4 child quads.
        """

        # Initialize the array that holds the child wuads.
        self._child_quads = list()
        
        # Left bottom coordinates of child quads
        # lt = left top, lb = left bottom, rt = right top, rb = right bottom.
        lt = [self._left_bottom[0], 
              self._left_bottom[1] + self._halve_size]
        lb = self._left_bottom
        rt = [self._left_bottom[0] + self._halve_size,
              self._left_bottom[1] + self._halve_size]
        rb = [self._left_bottom[0] + self._halve_size, 
              self._left_bottom[1]]

        # Add the quads
        self._child_quads.append(Quad(lt, self._halve_size,
                                 self._max_boddies,self))
        self._child_quads.append(Quad(lb, self._halve_size,
                                 self._max_boddies,self))
        self._child_quads.append(Quad(rt, self._halve_size,
                                 self._max_boddies,self))
        self._child_quads.append(Quad(rb, self._halve_size,
                                 self._max_boddies,self))

        # Current quad is not a leaf anymore
        self._leaf = False

        # Add the boddies of the current quad to the new child quads.
        for boddy in self._boddies:
            self.add_boddy(boddy) 
        
        # Empty the boddies that where hold by t his quad.
        self._boddies = None

    def _update_moment(self, mass = 0):
        """
            Update the multipole moment of the current quad
        In:
            praram: mass -- The mass to update the multipole moment with if the 
                            current quad is a leaf. 
        """

        # Current quad is a leaf, thus
        # add the mass (zeroth order multipole moment)
        if self._leaf:
            self._moment += mass
        # Current quad is not a leaf.
        else:
            # Reset the multipole moment and recalculate
            # it by looping of the child quads.
            self._moment = 0
            
            for quad in self._child_quads:
                self._moment += quad._moment 

        # Make sure the parents of the current quad update the multipole moment.
        if self._parent != None:
            self._parent._update_moment()


    def _plot(self, axis, quads):
        """
            Plot the current quad.
        In:
            param: axis -- The axis object to plot this quad at.
            param: qaudss -- A list of quads at which the current
                             quad should add a rectangle which represents
                             the current quad.
        """

        # Create the rectangle to plot of the current quad and add it to the list
        rect = patches.Rectangle(self._left_bottom, self._size, 
                                 self._size, fill=False)
        quads.append(rect)

        # If the current quad is a leaf plot the boddies.
        if self._leaf:
            if len(self._boddies) == 0:
                return
            a = np.array(self._boddies)
            axis.scatter(a[:,0],a[:,1],c='blue',s=1)
        # If the current quad is not a leaf recursively call this
        # method for all its 4 child quads.
        else:
            for quad in self._child_quads:
                quad._plot(axis, quads)



                 

    


    


        
        


