import numpy as np

def partition(lst, left, right, pivot_index):
    pivot = lst[pivot_index]
    lst[pivot_index] = lst[right]
    lst[right] = pivot
    store_index = left
    for i in range(left, right):
        if lst[i] < pivot:
            tmp = lst[store_index]
            lst[store_index] = lst[i]
            lst[i] = tmp
            store_index += 1
    tmp = lst[right]
    lst[right] = lst[store_index]
    lst[store_index] = tmp
    return store_index

def select(lst, left, right, k):
    if left == right:
        return lst[left]
    pivot_index = left
    pivot_index = partition(lst, left, right, pivot_index)
    if k == pivot_index:
        return lst[k]
    elif k < pivot_index:
        return select(lst, left, pivot_index - 1, k)
    else:
        return select(lst, pivot_index + 1, right, k)

if __name__ == "__main__":
    lst = [93, 6, 13, 2, 1]
    i = select(lst, 0, len(lst)-1, 2)
    print (i)
