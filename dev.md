# Dev

## View-dependent objectives

- Use the Spherical gaussian basis class for defining attention blending weights per prompt. For this case, basis weights should ABSOLUTELY be normalized.
- Need to add an xyz visualizer on the viewer, and test the round trip with the quaternion representation to make sure we can use it to define specific views.
- Dedfine prompt objectives using: -prompt -location (xyz, normalized if needed) -sigma

## More visualization

- Try visualizing weights of individual functions over the mesh for sanity checking

## Debug

- What the hell is going on with inverting subsequent light sources
- Figure out what's wrong with manual point_upwards on camera

## Camera adjustment

- Pan and move forwards/back camera to make object fit camera with set margin