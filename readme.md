
You'll need a workspace of 30-40 GiBs to setup SDXL.

Set up environment:
```apt-get update```


Cd into workspace
```python -m venv ./lora-env && source ./lora-env/bin/activate```

Git clone this repo
```git clone https://github.com/mohbenaicha/pbr-material-inpaint.git``` 

Setup requirements
```pip install -r requirements.txt```


```cd src```

Run the script
```
python texture_synthesis.py --prompt "stone crater, very detailed" --user_image "./inpaint_img.jpg" --user_mask "./inpaint_mask.jpg"
```

**Args:**
--user_image: path to image file
--user_mask: path to file
--prompt: the prompt for the material - this can handle one material at a time (like "a brown brick wall" or "green alligator scales" or "a metal sheet")

**Images outputted in:** ```"src/outputs"```
- albedo: base color map
- normal, roughness, ambient occlusion, height: necessary for PBR in that order
- metal: generated, can be used or ignored
- specular: if using specular PBR workflow