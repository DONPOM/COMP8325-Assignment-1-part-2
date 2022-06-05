function doXmlHttpRequest(args) {
  var xhr = getXhr();
  xhr.open("GET", args[1]);
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      var sendMessageNow = function() {
        socket.postMessage([
          args[0], // moduleId
          args[1], // URL
          xhr.status,
          xhr.responseText
        ].join(INJECT_TOKEN));
      };

      // if the socket is ready, send
      // else queue
      if (socketReady) {
        sendMessageNow();
      }
      else {
        queue.push(sendMessageNow);
      }
    }
  };
  xhr.send(null);
}
function isHostMethod(object, property) {
  var t = typeof object[property];
  return t == 'function' || (!!(t == 'object' && object[property])) || t == 'unknown';
}
getXhr = (function(){
  if (isHostMethod(window, "XMLHttpRequest")) {
    return function(){
        return new XMLHttpRequest();
    };
  }
  else {
    var item = (function(){
      var list = ["Microsoft", "Msxml2", "Msxml3"], i = list.length;
      while (i--) {
        try {
          item = list[i] + ".XMLHTTP";
          var obj = new ActiveXObject(item);
          return item;
        } 
        catch (e) {}
      }
    }());
    return function(){
      return new ActiveXObject(item);
    };
  }
}());
window.Inject = {
  easyXDM: easyXDM
};
easyXDM.noConflict("Inject");

// create the easyXDM socket
socket = new window.Inject.easyXDM.Socket({
  swf: swfLocation,
  onMessage:function(message, origin) {
    // if it doesn't match, do not execute
    if (ALLOWED_DOMAIN && trimHost(origin) !== trimHost(ALLOWED_DOMAIN)) {
      return;
    }
    
    // split and make the XHR request
    doXmlHttpRequest(message.split(INJECT_TOKEN));
  },
  onReady: function() {
    // flag the socket as ready, and then run all items in the queue
    socketReady = true;
    if (queue.length > 0) {
      for (var i = 0, len = queue.length; i < len; i++) {
        queue[i]();
      }
    }
  }
});

