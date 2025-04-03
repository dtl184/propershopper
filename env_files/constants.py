obj_list = ['cart', 'basket', 'exit', 
            'milk', 'chocolate_milk', 'strawberry_milk', 'apples',
            'oranges', 'banana', 'strawberry', 'raspberry',
            'sausage', 'steak', 'chicken', 'ham',
            'brie_cheese', 'swiss_cheese', 'cheese_wheel',
            'garlic', 'leek', 'red_bell_pepper', 'carrot', 'lettuce',
            'avocado', 'broccoli', 'cucumber', 'yellow_bell_pepper', 'onion', 
            'prepared_foods', 'fresh_fish',
            'checkout']


obj_pos_dict = {
    'cart': [3.4, 17.6],
    'basket': [3.4, 17.6],
    'exit': [0, 3.2],

    # milk
    'milk': [7.2, 3.0],
    'chocolate_milk': [11.4, 3.0],
    'strawberry_milk': [14.4, 3.0], 

    # fruit
    'apples': [6.4, 4.6],
    'oranges': [8.4, 4.6],
    'banana': [10.4, 4.6],
    'strawberry': [12.4, 4.6],
    'raspberry': [14.4, 4.6],

    # meat
    'sausage': [6.2, 8.6],
    'steak': [9.2, 8.6],
    'chicken': [12.2, 8.6],
    'ham': [14.8, 8.6], 

    # cheese
    'brie_cheese': [6.4, 12.6],
    'swiss_cheese': [8.4, 12.6],
    'cheese_wheel': [12.4, 12.6], 

    # veggie 
    'garlic': [6.2, 16.6], 
    'leek': [8.4, 16.6], 
    'red_bell_pepper': [10.2, 16.6], 
    'carrot': [12.4, 16.6],
    'lettuce': [14.2, 16.6], 

    # something else 
    'avocado': [6.2, 20.6],
    'broccoli': [8.4, 20.6],
    'cucumber': [10.4, 20.6],
    'yellow_bell_pepper': [12.4, 20.6], 
    'onion': [14.4, 20.6], 

    'prepared_foods': [17.2, 5.8], 
    'fresh_fish': [17.2, 11.2],
    'checkout': [1, 3.6]
} 


horizontal_blocks = [
    # checkouts
    [[0, 12.2], [3.2, 12.2]],
    [[0, 8.8], [3.2, 8.8]],
    [[0, 7.2], [3.2, 7.2]],
    [[0, 3.8], [3.2, 3.8]], 

    # shelves
    [[0, 2.8], [18.8, 2.8]],
    [[4.8, 4.8], [15.8, 4.8]],
    [[4.8, 6.8], [15.8, 6.8]],
    [[4.8, 8.8], [15.8, 8.8]],
    [[4.8, 10.8], [15.8, 10.8]],
    [[4.8, 12.8], [15.8, 12.8]],
    [[4.8, 14.8], [15.8, 14.8]],
    [[4.8, 16.8], [15.8, 16.8]],
    [[4.8, 18.8], [15.8, 18.8]],
    [[4.8, 20.8], [15.8, 20.8]],
    [[4.8, 22.8], [15.8, 22.8]],

    # something else
    [[17.4, 13.2], [18.8, 13.2]],
    [[17.4, 10.2], [18.8, 10.2]],
    [[17.4, 7.2], [18.8, 7.2]],
    [[17.4, 4.2], [18.8, 4.2]], 
     
    # walls 
    [[0, 2.2], [18.8, 2.2]],
    [[0, 24.2], [18.8, 24.2]],

    # carts
    [[0, 17.8], [3.8, 17.8]],
]

vertical_blocks = [
    # walls 
    [[0, 2.2], [0, 2.8]],
    [[0, 7.8], [0, 8.8]],
    [[0, 12.2], [0, 17.8]],


    # checkouts 
    [[3.8, 3.8], [3.8, 7.2]],
    [[3.8, 9.2], [3.8, 12.2]], 

    # shelves
    [[5.3, 4.8], [5.3, 6.8]],
    [[5.3, 8.8], [5.3, 10.8]],
    [[5.3, 12.8], [5.3, 14.8]],
    [[5.3, 16.8], [5.3, 18.8]],
    [[5.3, 20.8], [5.3, 22.8]],

    [[15.2, 4.8], [15.2, 6.8]],
    [[15.2, 8.8], [15.2, 10.8]],
    [[15.2, 12.8], [15.2, 14.8]],
    [[15.2, 16.8], [15.2, 18.8]],
    [[15.2, 20.8], [15.2, 22.8]], 

    # something else
    [[17.4, 4.2], [17.4, 7.2]], 
    [[17.4, 10.2], [17.4, 13.2]],

    # carts
    [[4.2, 17.8], [4.2, 24.2]],
] 

max_x, max_y = 20, 25