
function escapeString(in_str) {
  return in_str.replace(/</g, "&lt;")
               .replace(/>/g, "&gt;")
               .replace(/"/g, "&quot;")
               .replace(/'/g, "&#39;");
}

function isSubDomain(in_check, in_trust) {
  var t = in_trust.split('.');
  var c = in_check.split('.');
  if (t.length > c.length) {
    return false;
  }
  for (var i = 0; i < t.length; i++) {
    if (t[t.length - 1 - i] != c[c.length - 1 - i]) {
      return false;
    }
  }
  return true;
}

function isSafeUrl(in_url) {
  var e = in_url.split('/');
  if (e[0].indexOf(':') > 0) {
    if ((e[0].indexOf('http') == 0) || e[0].indexOf('https') == 0) {
      if (!e[2]) {
          return false;
      }
      var whitelist = [
        'yahoo.co.jp'
      ];
      for (var i = 0; i < whitelist.length; i++) {
        if (isSubDomain(e[2], whitelist[i])) {
            return true;
        }
      }
      return false;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

function parseQuery(in_url) {
  var ret = {};
  var tmp = in_url.split('?');
  if (tmp.length != 2) {
    return ret;
  }

  var params = tmp[1].split('&');
  for (var i = 0; i < params.length; i++) {
    var fldVal = params[i].split('=');
    if (fldVal.length == 1) {
      ret[fldVal[0]] = '';
    } else {
      ret[fldVal[0]] = escapeString(window.decodeURIComponent(fldVal[1]));
    }
  }
  return ret;
}

var gAdController = {
  objs : [],
  rank : 0,
  bind : function(callback, self) {
    return function() {
      callback.apply(self, arguments);
    };
  },
  isBeaconUrl : function (in_url) {
    var re = /^(https?:)?\/\/yadsbeacon\.yahoo\.co\.jp\/imps/;
    return re.test(in_url);
  },
  loadBeacon : function() {
    if (typeof(this.objs[this.rank]) == "undefined" || 
        !this.isBeaconUrl(this.objs[this.rank].ib_url)) {
      return;
    }

    var img = document.createElement('IMG');
    img.src = this.objs[this.rank].ib_url;
  },
  getAdElements : function(in_className) {
    if (document.getElementsByClassName) {
      return document.getElementsByClassName(in_className);
    } else {
      var ret = [];
      var cand = document.all ? document.all : document.getElementsByTagName("*");
      for (var i = 0; i < cand.length; i++) {
        if (cand.item(i).className == in_className) {
          ret.push(cand.item(i));
        }
      }
      return ret;
    }
  },
  needToErasePr : function() {
    var ydnImage = this.getAdElements('yads_ydn_frame_image');
    var ydnText  = this.getAdElements('yads_ydn_frame_text');

    if (ydnImage.length == 0) {
      if (ydnText.length == 0) {
        return false;
      }
      return true;
    } else {
      return false;
    }

  },
  switchPrTag : function() {
    var pr_elem = document.getElementById('yads-pr-label');
    if (!pr_elem || typeof(this.objs[this.rank]) == "undefined") {
      return;
    }
    
    if (this.objs[this.rank].pr_flag == '1') {
      pr_elem.style.display = 'block';
    } else {
      pr_elem.style.display = 'none';
      return;
    }
    
    if (this.needToErasePr()) {
      pr_elem.style.display = 'none';
    }
  },
  stringifyMessage : function(json) {
    switch (json.cmd) {
      case "eraseFrame" :
        return json.cmd + "," + json.frameSrc;
      case "resizeFrame":
        return json.cmd + "," + json.width + "," +  json.height + "," + json.frameSrc;
      default:
        return "";
    }
  },
  parseMessage : function(data) {
    var dataAry, cmd;
    dataAry = data.split(",");
    cmd = dataAry[0];

    switch (cmd) {
      case "eraseFrame" :
      return {cmd: 'eraseFrame', frameSrc: dataAry.slice(1).join()};
      case "resizeFrame":
        return {cmd: 'resizeFrame', width: dataAry[1], height: dataAry[2], frameSrc: dataAry.slice(3).join()};
      default:
        return {cmd:''};
    }
  },
  onMessageCallback : function (ev) {
    var obj, allowed, json;
    allowed = /yahoo\.co\.jp(:[0-9]+)?$/;
    if (!(ev.origin === 'http://i.yimg.jp' || allowed.test(ev.origin))) {
      return;
    }

    json = gAdController.parseMessage(ev.data);
    switch (json.cmd) {
      case 'eraseFrame' :
        gAdController.eraseFrame(json, ev.source);
        break;
      case 'resizeFrame':
        gAdController.resizeFrame(json, ev.source);
        break;
      default:
        break;
    }
  },
  resizeFrame : function(json, windowSrc) {
    var frame = this.findFrame(windowSrc);
    if (frame) {
      frame.width = json.width;
      frame.height = json.height;
    }
  },
  eraseFrame : function(json, windowSrc) {
    var frame = this.findFrame(windowSrc), noad_hook;
    if (frame) {
      frame.style.display = 'none';
      noad_hook = this.cache.yads_noad_hook[frame.id];
      yadsExecuteCallback(noad_hook, [{reason:2}]);
    }
  },
  findFrame : function(windowSrc) {
    var frames = document.getElementsByTagName('IFRAME');
    for (var i = 0; i < frames.length; i++) {
      if (windowSrc == frames[i].contentWindow) {
        return frames[i];
      }
    }
  },
  passback : function() {
    var new_obj;
    this.rank++;
    if (this.rank >= this.objs.length) {
      if (window != window.parent) {
        window.parent.postMessage(this.stringifyMessage({cmd:'eraseFrame', frameSrc:window.location.href}), "*");
      }
      return;
    }
    new_obj = this.objs[this.rank];
    if (window != window.parent) {
      if (typeof(new_obj.width) != "undefined" && typeof(new_obj.height) != "undefined") {
        window.parent.postMessage(this.stringifyMessage({cmd:'resizeFrame', frameSrc:window.location.href, width:new_obj.width, height:new_obj.height}), "*");
      }
    }
    document.write(new_obj.ad_tag);
  },
  adDone : function () {
    this.switchPrTag();
    this.loadBeacon();
  },
  start : function(in_objs) {
    var callback = this.bind(this.adDone, this);
    if (window.addEventListener) {
      window.addEventListener('load', callback, false);
    } else {
      if (window.attachEvent) {
        window.attachEvent('onload', callback);
      }
    }

    if (typeof(in_objs) == "undefined") {
      return null;
    }
    for (var i = 0; i < in_objs.length; i++) {
      this.objs[i] = in_objs[i];
    }

    document.write(this.objs[this.rank].ad_tag);
  },
  cache : (function() {
    if (gAdController && gAdController.cache) {
      return gAdController.cache;
    } else {
      return {};
    }
  })()
};

var gCRITEO = gAdController;

function yadsRequestAdTag(in_tagObjs) {
  gAdController.start(in_tagObjs);
}

var yadsRequestAdTag_v2 = yadsRequestAdTag;

var gAdTagCreater = {
  printScript : function() { 
    var url_strings = [];
    var url_params = parseQuery(location.href);
    var entry = (function(in_url) {
      if (typeof(in_url) == "undefined" || !isSafeUrl(in_url)) {
        return "";
      }
      return in_url;
    })(url_params['tag_path']);

    if (entry == "") {
      return;
    }

    for (var f in url_params) {
      if (f != 'tag_path') {
        var v = url_params[f];
        url_strings.push(f + '=' + window.encodeURIComponent(v));
      }
    }

    document.write('<script type="text/javascript" src="' + 
                   entry +
                   '?' +
                   url_strings.join('&') +
                   '"></scr' + 'ipt>' );
  },
  set : function() {
    this.printScript();
  }
};

