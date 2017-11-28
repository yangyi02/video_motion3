## Spatial Transformer Networks - 2 Frame 

- The simplest baseline that uses inverse warping
- Predict motion for every pixel
- Photometric loss for every pixel
- Input 2 frames
- Output is 1st frame
- Color image
- One more layer
- Add pairwise smoothness constraints

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01   |  |  |  | box, m_range=2, image_size=32, num_frame=3, bg_move |
| 02   |  |  |  | box_complex, m_range=2, image_size=32, num_frame=3, bg_move |
| 03   |  |  |  | mnist, m_range=2, image_size=32, num_frame=3, bg_move |
| 04   |  |  |  | robot64, m_range=2, image_size=64, num_frame=3 |
| 05   |  |  |  | mpii64, m_range=2, image_size=64, num_frame=3 |
| 07   |  |  |  | robot128, m_range=2, image_size=128, num_frame=3 |
| 08   |  |  |  | viper64, m_range=2, image_size=64, num_frame=3 |
| 09   |  |  |  | viper128, m_range=2, image_size=128, num_frame=3 |
| 10   |  |  |  | robot128c, m_range=2, image_size=128, num_frame=4 |

### Take Home Message

