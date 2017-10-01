## Predict Depth, Predict Appear Pixel Value - 4 Frame 

- Occlusion modeling, use neural net to predict discrete depth 
- Predict depth using only current frame with another network
- Predict motion for every pixel
- Photometric loss for every pixel
- Input 3 frames
- Output is 4th frame
- Color image
- Depth has 2 discrete level
- Predict pixel value for the new appearing location
- One more layer

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01   |  |  |  | box, m_range=2, image_size=32, num_frame=4, bg_move |
| 02   |  |  |  | box_complex, m_range=2, image_size=32, num_frame=4, bg_move |
| 03   |  |  |  | mnist, m_range=2, image_size=32, num_frame=4, bg_move |
| 04   |  |  |  | robot64, m_range=2, image_size=64, num_frame=4 |
| 05   |  |  |  | mpii64, m_range=2, image_size=64, num_frame=4 |
| 06   |  |  |  | nyuv2, m_range=2, image_size=64, num_frame=4 |

### Take Home Message

