import orjson


def iterate_on_songs(file_path):
    with open(file_path) as inp_file:
        for line in inp_file:
            data = orjson.loads(line)
            song = data['text']
            yield song
