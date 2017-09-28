## Predict Depth, Bidirectional Model - 5 Frame 

- Occlusion modeling, use neural net to predict discrete depth 
- Predict depth using only current frame with another network
- Predict motion for every pixel
- Photometric loss for every pixel
- Input 4 frames
- Output is 3rd frame
- Color image
- Depth has 2 discrete level
- Bidirectional model

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01   |  |  |  | box, m_range=2, image_size=32, num_frame=5, bg_move |
| 02   |  |  |  | box_complex, m_range=2, image_size=32, num_frame=5, bg_move |
| 03   |  |  |  | mnist, m_range=2, image_size=32, num_frame=5, bg_move |
| 04   |  |  |  | robot64, m_range=2, image_size=64, num_frame=5 |
| 05   |  |  |  | mpii64, m_range=2, image_size=64, num_frame=5 |
| 06   |  |  |  | nyuv2, m_range=2, image_size=64, num_frame=5 |

### Take Home Message

