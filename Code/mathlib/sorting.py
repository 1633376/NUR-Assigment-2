import numpy as np

def merge_sort(array):
    """
        Sort an array using merge sort

    In:
        param: array -- The array to sort.
    Out:
        return: The sorted array.
    """
    
    # Get the length of the array.
    size = len(array)
    
    # If the length is 1 then  return the input array.
    # This is an important check as this function is called
    # recursively. 
    if size == 1:
        return array

    # Split the array in an sorted left and right segment.
    left_sorted = merge_sort(array[:size >> 1])
    right_sorted = merge_sort(array[size >> 1:])

	#
    # Merge the left and right array.
	#    
    
    # The final sorted array.
    result = np.zeros(size)  
    
    # Current index in the left sorted array.
    left_idx = 0  
    # Current index in the right sorted array.
    right_idx = 0  
    # Current index in the final sorted array.
    result_idx = 0 
    
    # While we didn't fill the result array.
    while result_idx < size: 
               
        # Element from left array is smaller, insert it and increase position.
        if left_sorted[left_idx] < right_sorted[right_idx]: 
            result[result_idx] = left_sorted[left_idx]
            left_idx += 1
            result_idx += 1
        # Element from right array is smaller, insert it and increase position.
        else:  
            result[result_idx] = right_sorted[right_idx]
            right_idx += 1
            result_idx += 1
        
        # Only right arrat has elements left, insert the remaining elements.
        if left_idx == len(left_sorted): 
            result[result_idx:] = right_sorted[right_idx:]
            break
            
        # Only left has elements left, insert the remaining elements
        if right_idx == len(right_sorted): 
            result[result_idx:] = left_sorted[left_idx:]
            break
            
    return result