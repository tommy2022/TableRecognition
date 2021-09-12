# Table Recognizer
## Why
Swimmer records are taken during practices to track improvements over periods of time. Traditionally, pen was used on paper with pre-printed grid (see images/gridded for examples). Some times when pre-printed grids are not available, grid are either drawn by hand or is not drawn at all (See images/hand_drawn). These then need to be manually inputted into a spreadsheet to compute averages and other statistical measures. This is difficult to do without enough staffs resulting in clubs to skip this process. At the club that I was at, we typically only had digital processed data during training camps that we have twice a year. These data are often useful for swimmers to assess their progress, especailly when they try out new forms (new swimming techniques).

Recently, there have been some shift in using ipads or other tablets to record in place of pen and paper to eliminate the manual input of data. However, this is harder to record and often requires one staff/coach dedicated specifically to recording down the swimmer's records. This makes it hard for many smaller clubs to use tablets to record.

One solution to this is to implement a program that takes in image file and outputs csv files to be imported into a spreadsheet.

## How
The project was broken down into three parts. 
1. Recognizing tables with pre-printed grids (Cell and character segmentation)
2. Recognizing table without pre-printed grids (Cell and character segmentation)
3. Classifying numbers and create the csv output.

### Preprocessing
1. Sauvola Binarization
2. Four point transform on the largest rectangle to crop out the desired area

### Part 1
1. Hough line transform to get the vertical and horizontal lines.
2. Kmeans to group the vertical and horizontal lines.
3. Because multiple hough lines were detected for each line, hough lines on the same lines were grouped based on the difference in differences of y-intercepts (or x-intercepts for vertical lines). The longest line from each group was selected.
4. Remove the grid using bit-masking
5. Apply character segmentation on each table cell


## TODO
- Implement Digit Recognizer to recognize the numbers. LSTM could work since records will be similar to previous.
- Implement table recognizer for table without printed grid.