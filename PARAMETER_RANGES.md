# 

|Parameter|Documented candidate domain|
|-|-|
|SD learning rate|1e-6 to 1e-4, step 1e-6|
|VAE learning rate|1e-4 to 1e-2, step 1e-4|
|SD training epochs|5 to 500, step 5|
|SD diffusion steps|3 to 198, step 5|
|VAE latent dimension|20 to 1000, step 10|
|VAE hidden dimension|latent dimension to 1000, step 50 from the chosen latent dimension|
|Batch size|30 to 1000, step 10|
|SD hidden layers|integers 0 to 5|
|Mean auxiliary edges per user|log-scale 0.25 to 12; lower bound 1 when a minimum of one edge is required|
|Activity / quality allocation strength|0 to 2 / 0 to 3|
|Initial SD score mixing coefficient|log-scale 0.01 to 0.9|
|Auxiliary-edge weight|log-scale 0.5 to 4|



