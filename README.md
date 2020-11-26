# Learning-to-Detect-Pedestrian-Flow-inTraffic-Intersections-from-Synthetic-Data

<h6>
 The goal of this research project was to detect pedestrian flow at the traffic intersection. As there was no available dataset to accomplish the task, we have created a synthetic pedestrian dataset using GTA V video game.
  
<h1> Proposed Framework  
 <h6> 
  <img src="https://github.com/abhijit-buet/Images/blob/main/Summary.PNG" width="512" height = "350">
   
   
   Our proposed PedestrianNet is a two branch CNN network where we have used AlexNet as feature extractor.
   We required two branch as the input consists of image frames and their corresponding optical flow.
   
   <h1> PedestrianNet  
  <h6>
     <img src="https://github.com/abhijit-buet/Images/blob/main/AlexNet.PNG" width="512" height = "350">
    
 <h1> Result
  <h6>
    
   Here, we have tried to detect pedestrian flow in three different direction - from left to right, from right to left and combined number of pedestrian. We have compared our result with YOLO V3 as a human detector. Mean Absolute Error(MAE) and Mean Square Error(MSE) were used as evaluation metric.
   
   <img src="https://github.com/abhijit-buet/Images/blob/main/Capture.PNG" width="512" height = "232">
