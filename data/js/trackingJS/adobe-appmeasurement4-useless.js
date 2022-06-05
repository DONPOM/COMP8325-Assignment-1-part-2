

// adbreimaginetest analytics tool
if (_satellite.getVar('useTool_adbreimaginetest')) {
    _satellite.notify('Using tool adbreimaginetest', 1);
} else {
    _satellite.tools['ba1811d4769e557169de925c6cbb9b6b'].settings.initTool = false;
    _satellite.notify('Not using tool adbreimaginetest', 1);
}

