from __init__ import FileWriter, FileReader

writer = FileWriter('./data', 'test')
writer.init_file()
for i in range(10):
    writer.write_val(i)

reader = FileReader('./data', 'test')
vals = reader()
assert vals['score'] == [i for i in range(10)]
