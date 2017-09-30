## Predict Depth, Predict Appear Pixel Value - 3 Frame 

- Occlusion modeling, use neural net to predict discrete depth 
- Predict depth using only current frame with another network
- Predict motion for every pixel
- Photometric loss for every pixel
- Input 2 frames
- Output is 3rd frame
- Color image
- Depth has 2 discrete level
- Predict pixel value for the new appearing location
- Motion range is 4

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01 |  |  |  | box, m_range=4, image_size=32, num_frame=3, bg_move |
| 01-1 |  |  |  | box_complex, m_range=4, image_size=32, num_frame=3, bg_move |
| 02 |  |  |  | mnist, m_range=4, image_size=32, num_frame=3, bg_move |
| 03 |  |  |  | robot64, m_range=4, image_size=64, num_frame=3 |
| 04 |  |  |  | mpii64, m_range=4, image_size=64, num_frame=3 |
| 05 |  |  |  | nyuv2, m_range=4, image_size=64, num_frame=3 |

### Take Home Message

