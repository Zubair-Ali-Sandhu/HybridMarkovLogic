# Python script to generate colorful code image
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

code = '''static int MP4_ReadBox_sample_vide( stream_t *p_stream, MP4_Box_t *p_box )
{
    uint32_t i_sample_size;

    // [code omitted for brevity]

    i_sample_size = i_read;  
    p_box->data.p_sample_vide->p_qt_image_description = malloc(i_sample_size);

    // [code omitted for brevity]
}'''

fig, ax = plt.subplots(figsize=(10, 6))
ax.text(0.05, 0.95, code, family='monospace', fontsize=12,
        verticalalignment='top', color='white',
        backgroundcolor='black', transform=ax.transAxes)
ax.axis('off')
plt.savefig('vlc_vulnerable_code.png', bbox_inches='tight', dpi=300)
plt.close()

print("Image with vulnerable code saved as 'vlc_vulnerable_code.png'")