# goselo_caffe

Python code to try GOSELO.

Asako Kanezaki, Jirou Nitta and Yoko Sasaki.
**GOSELO: Goal-Directed Obstacle and Self-Location Map for Robot Navigation using Reactive Neural Networks.** 
*IEEE Robotics and Automation Letters (RA-L)*, Vol.3, Issue 2, pp.696-703, 2018. (**presented in ICRA'18**)

([pdf](https://doi.org/10.1109/LRA.2017.2783400))
([project](https://kanezaki.github.io/goselo/))

### Requirement

Install [caffe](http://caffe.berkeleyvision.org/).  
Prepare your Makefile.config and compile.  

    $ make; make pycaffe

### Download

    $ wget https://data.airc.aist.go.jp/kanezaki.asako/pretrained_models/goselo_visible.caffemodel
    $ wget https://data.airc.aist.go.jp/kanezaki.asako/pretrained_models/goselo_invisible.caffemodel

### Demo

    $ python demo_navigation_visible.py map_images/default.png
or  

    $ python demo_navigation_invisible.py map_images/default.png

