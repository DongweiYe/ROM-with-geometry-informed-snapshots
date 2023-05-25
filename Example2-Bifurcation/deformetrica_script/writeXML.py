import numpy as np

text_file = open("deformetrica_script/data_set.xml", "w")

text_file.write('<?xml version="1.0"?>\n')
text_file.write('<data-set>\n')
text_file.write('\n')

for i in range(0,500):
	dirt = '../data/raw/sample_'+str(i)+'/'
	text_file.write('    <subject id="sample'+str(i)+'">\n')
	text_file.write('        <visit id="t0">\n')
	text_file.write('            <filename object_id="mesh">'+dirt+'boundary.vtk</filename>\n')
	text_file.write('        </visit>\n')
	text_file.write('    </subject>\n')
	text_file.write('\n')

text_file.write('</data-set>')
text_file.close()
	