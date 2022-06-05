var loadTimerStart = new Date().valueOf();
		var _EXPERIMENTID =null;
		var _PREVIEW =false;

		var _ISALPHASERVER =false;
		var _ISTESTSERVER =false;

		var _BROWSER = 'Safari';
		var _PLATFORM = 'Mac';(function(e){var c=e.m$portal;c||(c=e.m$portal={});c.cookie=function(c,b,a){if(void 0!==b){var d,a=a||{};null===b&&(b="",a.expires=-1);b+="";document.cookie=c+"="+b+(a.expires&&(d="number"==typeof a.expires&&(d=new Date),d.setTime(d.getTime()+864E5*a.expires),d||"toUTCString"in a.expires&&a.expires)&&"; expires="+d.toUTCString()||"")+(a.path?"; path="+a.path:"")+(a.domain?"; domain="+a.domain:"")+(a.secure?"; secure":"");return b}if(""!==(document.cookie||""))return b=(document.cookie.match(RegExp("(?:^| )"+
c+"\\=(\\S*)(?:; |$)"))||[])[1],void 0===b?void 0:b}})(window);
if (!Function.prototype.bind) {
	Function.prototype.bind = function bind(that) {
		var target = this;

		return function(){
			return target.apply(that, arguments);
		};
	}
}

if (!Array.isArray){
	Array.isArray = function(obj) {
		return '' + obj !== obj && Object.prototype.toString.call(obj) == '[object Array]';
	}
}

if(!Array.prototype.indexOf){
	Array.prototype.indexOf = function(obj){
		for(var i=0; i<this.length; i++){
			if(this[i]==obj){
				return i;
			}
		}
		return -1;
	}
}

function extend(target, source, deep){
	target = target || {};
	for (var p in source)if (source.hasOwnProperty(p) && !(p in target)){
		target[p] = source[p];
	}
	return target;
}

var Splash = {
		ajaxUpdates: [],
		data: {},
		tpl: {}
	};

document.documentElement.id = 'jsHtml';
document.documentElement.className += ' jsHtml';

try { top.location.toString(); } catch (er) { top['location'] = location; }
if (navigator.userAgent.indexOf('MSIE 6') !== -1) {
	try {
		document.execCommand('BackgroundImageCache', false, true);
	} catch(e){}
}

var mr = {
	id: function(id) {
		return document.getElementById(id);
	},
	hasClass: function(elem, cls) {
		if (elem.nodeType === 1 && (" " + elem.className + " ").replace(/[\n\t\r]/g, " ").indexOf(" " + cls + " ") > -1) {
			return true;
		}
	},
	addClass: function(elem, cls) {
		if (!mr.hasClass(elem, cls))
			elem.className += ' ' + cls;
	},
	removeClass: function(elem, cls) {
		elem.className = elem.className.replace(new RegExp('\\b' + cls + '\\b'), '', 'g');
	},
	bind: function(elem, event, func) {
		if (elem.addEventListener){
			elem.addEventListener(event, func, false);
		} else if (elem.attachEvent){
			elem.attachEvent('on' + event, func);
		}
	},
	unbind: function(elem, event, func) {
		if (elem.removeEventListener){
			elem.removeEventListener(event, func, false);
		} else if (elem.detachEvent){
			elem.detachEvent('on' + event, func);
		}
	},
	cookie: function(name, value, domain) {
		var options = {
			path: '/',
			expires: 365
		};

		domain && (options.domain = domain);

		return __PH.cookie(name, value, options);
	},
	getScript: function(src, callback, charset) {
		var script = document.createElement('script');
		if (charset) {
			script.charset = charset;
		}

		if (callback) {
			if (typeof script.onreadystatechange !== 'undefined') {
				script.onreadystatechange = function() {
					if (script.readyState === 'loaded' || script.readyState === 'complete') {
						script.onloadDone = true;
						callback();
					}
				};
			} else {
				script.onload = callback;
			}
		}

		script.src = src;
		document.documentElement.firstChild.appendChild(script);
	},
	counter: function(id) {
		if (!isNaN(id)){
			id = 'cln' + id;
		}
		window.rb_counter(id);
	},
	linearBezier: function(q,w,e,r){function b(d,g,i,h){this.b={x:d,y:g};this.c={x:i,y:h}}function c(d,g,i){var h=3*g,g=3*(i-g)-h;
		return(((1-h-g)*d+g)*d+h)*d}b.prototype.get=function(d){var g=this;if(0==d||1==d)return{x:d,y:d};if(!(0<d)||!(1>d))throw new RangeError;
		return{x:c(d,g.b.x,g.c.x),y:c(d,g.b.y,g.c.y)}};return new b(q,w,e,r)},

	getDomain: function(url){
		return (url || window.location.href).replace(/https?:\/\//i, '').split('/')[0];
	}
};

extend(mr, {extend: extend});

var imagesPreloader = new function(){
	this._images = {};
	this.add = function(src, img){
		this._images[src] = img;
	};
	this.load = function(){
		for (var src in this._images) if (this._images.hasOwnProperty(src)){
			var img = this._images[src];
			if (!img){
				new Image().src = src;
			} else {
				(function(img, src){
					setTimeout(function(){
						img.src = src;
					}, 0);
				})(img, src);
			}
		}
	};
};

var callbackQuery = new function(){
	var _cb = [],
		loaded = false;

	this.run = function(cb){
		var args = Array.prototype.slice.call(arguments, 1);
		if (loaded){
			cb(args);
		} else {
			_cb.push({
				cb: cb,
				args: args
			});
		}
	};

	this.loaded = function(){
		loaded = true;
		for (var i = 0, l = _cb.length; i < l; i++){
			var cb = _cb[i];
			cb.cb(cb.args);
		}
		_cb = null;
	}
};

function _plural(n, L, i){
// plural
var x = ((n%100 <= 10) || (n%100 >= 20)) ? n % 10 : 0, l = L.length;
if( l == 2 && x == 1 )	x = 0;
else if( l == 3 && x > 1 ) x = x < 5 ? 2 : 0;
return (i ? n + (i && (i!=1) ? i : '') : '') + (typeof(L) != 'string' ? (L[x] ||  L[(x > 1 && x < 5) ? x : 0] || L[x > 1?1:0]) : L);
}

mr.hash = window.location.href;
mr.hash = mr.hash.indexOf('#') === -1 ? null : mr.hash.substr(mr.hash.indexOf('#') + 1);
mr.domain = mr.getDomain();

function logError(e, text, throwError){
	function buildRegExp(list) {
		return new RegExp("(" + list.join("|") + ")", "g");
	}

	var ignored = {
		message: buildRegExp([
			  'uplListener'
			, 'jsbSmile'
			, 'dpQuery'
			, 'DealPly'
		])
	};

	if(!e.message || e.message.match(ignored.message)) {
		return;
	}

	var name = 'JSError' + (text ? '_' + text : '');
	createRadar('error')('all',0,1)('all',1,2)(name,0,1)(name,1,2)(true);

	if (mr.logErrors){
		var log = [
			  'splashErrors'
			, encodeURIComponent(e.url || window.location.href)
			, e.line || 0
			, encodeURIComponent(
				  (mr.logErrorsExtended ? 'region=' + _EFFREGIONID
				+ '; browser=' + _BROWSER
				+ '; platform=' + _PLATFORM
				+ '; error=' : '') + (e.name ? e.name + ': ' : '') + e.message
				+ (e.stack ? '; stack=' + e.stack.replace(/\n+/g, ', ').replace(/\s+/ig, ' ') : '')
			)
			, encodeURIComponent(window.location.href)
			, 'mmain'
		].join('|');

		(new Image()).src = '//gstat.imgsmail.ru/gstat'
			+ '?ua=1'
			+ '&clienterror.mmain=1'
			+ '&logme=' + log
			+ '&rnd=' + (new Date()).getTime() + Math.random()
		;
	}

	if (throwError !== false && !e._r && (_ISALPHASERVER || _ISTESTSERVER)){
		e._r = true;
		throw e;
	}
}

window.onerror = function(message, url, line){
	var domain = url && mr.getDomain(url);

	if (domain && (domain == 'limg.imgsmail.ru' || domain == 'img.imgsmail.ru' || domain == 'portal.mail.ru'  || domain == 'auth.mail.ru' || (domain == mr.domain && mr.logInlineErrors)) || !domain){
		logError({message: message, url: url || 'undefined url', line: line}, 'onerror' + (domain == mr.domain ? 'Inline' : ''), false);
	}
};

function getRegionInfo(id, parentId, grandId, hasChildren){
	var currentLevel = 3,
		level1 = grandId || null,
		level2 = parentId || null,
		level3 = id || null,
		levels = 3,
		parentNotEmpty;

	if (!level2){
		level1 = id;
		level2 = level3 = null;
		currentLevel = 1;
		levels = 1;

	} else if (!level1){
		level1 = level2;
		level2 = level3;
		level3 = null;
		currentLevel = 2;
		if (!hasChildren){
			levels = 2;
		} else {
			levels = 3;
		}
	}

	return {
		levels: levels,
		currentLevel: currentLevel,
		level1: level1,
		level2: level2,
		level3: level3,
		id: id,
		parentId: parentId,
		grandId: grandId
	};
};

var percent = (function(date){
	var percent = date.substring(date.length - 2);

	return function(p){
		return percent < p;
	};
})(new Date*1+'');
var _EFFREGIONID = '217';

		var region = getRegionInfo(217,null,null,1);

		var effRegion = getRegionInfo(217,null,null,1);

		mr.activeEmail = '';
		mr.timezone =39600;
		mr.referrer = '';
		var _timezone = mr.timezone;

		mr.timestamp =1417582788610;

		mr.dpr = window.devicePixelRatio;
		mr.retina = document.createElement('div').style.backgroundSize != void 0 && mr.dpr > 1;

		if (mr.retina){
			mr.removeClass(document.getElementsByTagName('html')[0], 'nonRetina');
			mr.addClass(document.getElementsByTagName('html')[0], 'retina');
		}

		mr.logErrors =true;
		mr.logErrorsExtended =true;
		mr.logInlineErrors =true;
		mr.LF1Percent =5;