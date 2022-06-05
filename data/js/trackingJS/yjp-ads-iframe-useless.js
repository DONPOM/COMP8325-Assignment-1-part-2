

gAdTagCreater.set();

window.onload = function() { 
  var iframes = document.getElementsByTagName('iframe'); 
  if(iframes.length === 1) { 
    var w = iframes[0].getAttribute('width');
    var rw = w.match(/^([0-9]+)(px)?/);
    var mw = parseInt(rw[1]);
    var h = iframes[0].getAttribute('height');
    var rh = h.match(/^([0-9]+)(px)?/);
    var mh = parseInt(rh[1]) + 200;
    iframes[0].setAttribute('width','100%'); 
    iframes[0].setAttribute('height','100%');
    iframes[0].style.minWidth = mw + "px";
    iframes[0].style.minHeight = mh + "px";
  } 
}

