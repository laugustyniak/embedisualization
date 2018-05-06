from embedisualization.embedisualization import Embedisualisation

if __name__ == '__main__':
    emb = Embedisualisation([
        'This is example. This is subfirst.', 'this is next example', 'this is third example',
        'This is task. This is subfirst.', 'this is next tesk', 'this is third tesk',
    ])
    emb.create_d3_visualisation()
