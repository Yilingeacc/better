def generate_item2type():
    type2id = {}
    with open('tag2id.txt', 'r') as f:
        pairs = f.read().split('\n')
        for pair in pairs:
            type, id = pair.split(':')
            type2id[type] = id

    item2id = {}
    with open('../data/MovieLens-Rand/movies.dat', 'r') as f:
        movies = f.read().split('\n')
        for movie in movies:
            id, name, types = movie.split('::')
            types = types.split('|')
            typeids = []
            for type in types:
                typeids.append(type2id[type])
            item2id[id] = typeids

    with open('id2typeid.txt', 'w') as f:
        for k, v in item2id.items():
            type_str = ",".join(v)
            f.write(f'{k}:{type_str}\n')


generate_item2type()
