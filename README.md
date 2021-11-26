# Learning-to-Detect-Pedestrian-Flow-inTraffic-Intersections-from-Synthetic-Data

<h6>
 The goal of this research project was to detect pedestrian flow at the traffic intersection. As there was no available dataset to accomplish the task, I created a synthetic pedestrian dataset using GTA V video game. 
 Then I performed domain adaptaion using Cycle GAN, translating the synthetic images to photo-realistic images. In the end, I finetuned the Pedestrian Counting Net (PCNet) using a small amount of real data.
 <h1> Dataset
  <h6>
   The Synthetic Traffic-intersection Pedestrian Dataset will be released soon.
<h1> Proposed Framework  
 <h6> 
  <img src="https://github.com/abhijit-buet/Learning-to-Detect-Pedestrian-Flow-inTraffic-Intersections-from-Synthetic-Data/blob/main/frame.PNG" width="768" height = "350">
   
   
   The proposed Pedestrian Counting Net is a two branch CNN framework.
   The reason behind using two branch is that the input consists of image frames and their corresponding optical flow.
   
   <h1> PedestrianFlowNet  
  <h6>
     <img src="https://github.com/abhijit-buet/Images/blob/main/AlexNet.PNG" width="768" height = "350">
    
 <h1> Result
  <h6>
    
   Here, The PCNet detects pedestrian flow in three different direction - from left to right, from right to left and combined number of pedestrian. I compared the result with YOLO V3 as a human detector. Mean Absolute Error(MAE) and Mean Square Error(MSE) were used as evaluation metric.
   
   <img src="https://github.com/abhijit-buet/Learning-to-Detect-Pedestrian-Flow-inTraffic-Intersections-from-Synthetic-Data/blob/main/result.PNG" width="768" height = "232">
