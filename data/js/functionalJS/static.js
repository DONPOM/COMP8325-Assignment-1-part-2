function apprise(string, args, callback)
{
    var default_args = {
        'confirm': false,
        'verify': false,
        'input': false,
        'animate': false,
        'textOk': 'Ok',
        'textCancel': 'Cancel',
        'textYes': 'Yes',
        'textNo': 'No'
    }
    if (args)
    {
        for (var index in default_args)
        {
            if (typeof args[index] == "undefined")
                args[index] = default_args[index];
        }
    }
    var aHeight = $(document).height();
    var aWidth = $(document).width();
    $('body').append('<div class="appriseOverlay mainShare" id="aOverlay"></div>');
    $('.appriseOverlay').css('height', aHeight).css('width', aWidth).fadeIn(100);
    $('body').append('<div class="appriseOuter mainShare"></div>');
    $('.appriseOuter').append('<div class="appriseInner mainShare"></div>');
    $('.appriseInner').append(string);
    $('.appriseOuter').css("left", ($(window).width() - $('.appriseOuter').width()) / 2 + $(window).scrollLeft() + "px");
    if (args)
    {
        if (args['animate'])
        {
            var aniSpeed = args['animate'];
            if (isNaN(aniSpeed)) {
                aniSpeed = 400;
            }
            $('.appriseOuter').css('top', '-200px').show().animate({
                top: "100px"
            }, aniSpeed);
        } else
        {
            $('.appriseOuter').css('top', '140px').fadeIn(200);
        }
    } else
    {
        $('.appriseOuter').css('top', '140px').fadeIn(200);
    }
    if (args)
    {
        if (args['input'])
        {
            if (typeof(args['input']) == 'string')
            {
                $('.appriseInner').append('<div class="aInput mainShare"><input type="text" class="aTextbox mainShare" t="aTextbox" value="' + args['input'] + '" /></div>');
            } else
            {
                $('.appriseInner').append('<div class="aInput mainShare"><input type="text" class="aTextbox mainShare" t="aTextbox" /></div>');
            }
            $('.aTextbox').focus();
        }
    }
    $('.appriseInner').append('<div class="aButtons mainShare"></div>');
    if (args)
    {
        if (args['confirm'])
        {
            $('.aButtons').append('<button class="mainShare" value="ok">' + args['textOk'] + '</button>');
            $('.aButtons').append('<button value="cancel">' + args['textCancel'] + '</button>');
        } else if (args['verify'])
        {
            $('.aButtons').append('<button class="mainShare" value="ok">' + args['textYes'] + '</button>');
            $('.aButtons').append('<button value="cancel">' + args['textNo'] + '</button>');
        } else
        {
            $('.aButtons').append('<button class="mainShare" value="ok">' + args['textOk'] + '</button>');
        }
    } else
    {
        $('.aButtons').append('<button class="mainShare" value="ok">Ok</button>');
    }
    $(document).keydown(function(e)
    {
        if ($('.appriseOverlay').is(':visible'))
        {
            if (e.keyCode == 13)
            {
                $('.aButtons > button[value="ok"]').click();
            }
            if (e.keyCode == 27)
            {
                $('.aButtons > button[value="cancel"]').click();
            }
        }
    });
    var aText = $('.aTextbox').val();
    if (!aText) {
        aText = false;
    }
    $('.aTextbox').keyup(function()
    {
        aText = $(this).val();
    });
    $('.aButtons > button').click(function()
    {
        $('.appriseOverlay').remove();
        $('.appriseOuter').remove();
        if (callback)
        {
            var wButton = $(this).attr("value");
            if (wButton == 'ok')
            {
                if (args)
                {
                    if (args['input'])
                    {
                        callback(aText);
                    } else
                    {
                        callback(true);
                    }
                } else
                {
                    callback(true);
                }
            } else if (wButton == 'cancel')
            {
                callback(false);
            }
        }
    });
}
jQuery.easing['jswing'] = jQuery.easing['swing'];
jQuery.extend(jQuery.easing, {
    def: 'easeOutQuad',
    swing: function(x, t, b, c, d) {
        return jQuery.easing[jQuery.easing.def](x, t, b, c, d);
    },
    easeInQuad: function(x, t, b, c, d) {
        return c * (t/=d) * t + b;
    },
    easeOutQuad: function(x, t, b, c, d) {
        return - c * (t/=d) * (t - 2) + b;
    },
    easeInOutQuad: function(x, t, b, c, d) {
        if ((t/=d / 2) < 1)
            return c / 2 * t * t + b;
        return - c / 2 * ((--t) * (t - 2) - 1) + b;
    },
    easeInCubic: function(x, t, b, c, d) {
        return c * (t/=d) * t * t + b;
    },
    easeOutCubic: function(x, t, b, c, d) {
        return c * ((t = t / d - 1) * t * t + 1) + b;
    },
    easeInOutCubic: function(x, t, b, c, d) {
        if ((t/=d / 2) < 1)
            return c / 2 * t * t * t + b;
        return c / 2 * ((t -= 2) * t * t + 2) + b;
    },
    easeInQuart: function(x, t, b, c, d) {
        return c * (t/=d) * t * t * t + b;
    },
    easeOutQuart: function(x, t, b, c, d) {
        return - c * ((t = t / d - 1) * t * t * t - 1) + b;
    },
    easeInOutQuart: function(x, t, b, c, d) {
        if ((t/=d / 2) < 1)
            return c / 2 * t * t * t * t + b;
        return - c / 2 * ((t -= 2) * t * t * t - 2) + b;
    },
    easeInQuint: function(x, t, b, c, d) {
        return c * (t/=d) * t * t * t * t + b;
    },
    easeOutQuint: function(x, t, b, c, d) {
        return c * ((t = t / d - 1) * t * t * t * t + 1) + b;
    },
    easeInOutQuint: function(x, t, b, c, d) {
        if ((t/=d / 2) < 1)
            return c / 2 * t * t * t * t * t + b;
        return c / 2 * ((t -= 2) * t * t * t * t + 2) + b;
    },
    easeInSine: function(x, t, b, c, d) {
        return - c * Math.cos(t / d * (Math.PI / 2)) + c + b;
    },
    easeOutSine: function(x, t, b, c, d) {
        return c * Math.sin(t / d * (Math.PI / 2)) + b;
    },
    easeInOutSine: function(x, t, b, c, d) {
        return - c / 2 * (Math.cos(Math.PI * t / d) - 1) + b;
    },
    easeInExpo: function(x, t, b, c, d) {
        return (t == 0) ? b : c * Math.pow(2, 10 * (t / d - 1)) + b;
    },
    easeOutExpo: function(x, t, b, c, d) {
        return (t == d) ? b + c : c * ( - Math.pow(2, - 10 * t / d) + 1) + b;
    },
    easeInOutExpo: function(x, t, b, c, d) {
        if (t == 0)
            return b;
        if (t == d)
            return b + c;
        if ((t/=d / 2) < 1)
            return c / 2 * Math.pow(2, 10 * (t - 1)) + b;
        return c / 2 * ( - Math.pow(2, - 10*--t) + 2) + b;
    },
    easeInCirc: function(x, t, b, c, d) {
        return - c * (Math.sqrt(1 - (t/=d) * t) - 1) + b;
    },
    easeOutCirc: function(x, t, b, c, d) {
        return c * Math.sqrt(1 - (t = t / d - 1) * t) + b;
    },
    easeInOutCirc: function(x, t, b, c, d) {
        if ((t/=d / 2) < 1)
            return - c / 2 * (Math.sqrt(1 - t * t) - 1) + b;
        return c / 2 * (Math.sqrt(1 - (t -= 2) * t) + 1) + b;
    },
    easeInElastic: function(x, t, b, c, d) {
        var s = 1.70158;
        var p = 0;
        var a = c;
        if (t == 0)
            return b;
        if ((t/=d) == 1)
            return b + c;
        if (!p)
            p = d * .3;
        if (a < Math.abs(c)) {
            a = c;
            var s = p / 4;
        } else 
            var s = p / (2 * Math.PI) * Math.asin(c / a);
        return - (a * Math.pow(2, 10 * (t -= 1)) * Math.sin((t * d - s) * (2 * Math.PI) / p)) + b;
    },
    easeOutElastic: function(x, t, b, c, d) {
        var s = 1.70158;
        var p = 0;
        var a = c;
        if (t == 0)
            return b;
        if ((t/=d) == 1)
            return b + c;
        if (!p)
            p = d * .3;
        if (a < Math.abs(c)) {
            a = c;
            var s = p / 4;
        } else 
            var s = p / (2 * Math.PI) * Math.asin(c / a);
        return a * Math.pow(2, - 10 * t) * Math.sin((t * d - s) * (2 * Math.PI) / p) + c + b;
    },
    easeInOutElastic: function(x, t, b, c, d) {
        var s = 1.70158;
        var p = 0;
        var a = c;
        if (t == 0)
            return b;
        if ((t/=d / 2) == 2)
            return b + c;
        if (!p)
            p = d * (.3 * 1.5);
        if (a < Math.abs(c)) {
            a = c;
            var s = p / 4;
        } else 
            var s = p / (2 * Math.PI) * Math.asin(c / a);
        if (t < 1)
            return - .5 * (a * Math.pow(2, 10 * (t -= 1)) * Math.sin((t * d - s) * (2 * Math.PI) / p)) + b;
        return a * Math.pow(2, - 10 * (t -= 1)) * Math.sin((t * d - s) * (2 * Math.PI) / p) * .5 + c + b;
    },
    easeInBack: function(x, t, b, c, d, s) {
        if (s == undefined)
            s = 1.70158;
        return c * (t/=d) * t * ((s + 1) * t - s) + b;
    },
    easeOutBack: function(x, t, b, c, d, s) {
        if (s == undefined)
            s = 1.70158;
        return c * ((t = t / d - 1) * t * ((s + 1) * t + s) + 1) + b;
    },
    easeInOutBack: function(x, t, b, c, d, s) {
        if (s == undefined)
            s = 1.70158;
        if ((t/=d / 2) < 1)
            return c / 2 * (t * t * (((s*=(1.525)) + 1) * t - s)) + b;
        return c / 2 * ((t -= 2) * t * (((s*=(1.525)) + 1) * t + s) + 2) + b;
    },
    easeInBounce: function(x, t, b, c, d) {
        return c - jQuery.easing.easeOutBounce(x, d - t, 0, c, d) + b;
    },
    easeOutBounce: function(x, t, b, c, d) {
        if ((t/=d) < (1 / 2.75)) {
            return c * (7.5625 * t * t) + b;
        } else if (t < (2 / 2.75)) {
            return c * (7.5625 * (t -= (1.5 / 2.75)) * t + .75) + b;
        } else if (t < (2.5 / 2.75)) {
            return c * (7.5625 * (t -= (2.25 / 2.75)) * t + .9375) + b;
        } else {
            return c * (7.5625 * (t -= (2.625 / 2.75)) * t + .984375) + b;
        }
    },
    easeInOutBounce: function(x, t, b, c, d) {
        if (t < d / 2)
            return jQuery.easing.easeInBounce(x, t * 2, 0, c, d) * .5 + b;
        return jQuery.easing.easeOutBounce(x, t * 2 - d, 0, c, d) * .5 + c * .5 + b;
    }
});
jQuery.extend(jQuery.easing, {
    easeIn: function(x, t, b, c, d) {
        return jQuery.easing.easeInQuad(x, t, b, c, d);
    },
    easeOut: function(x, t, b, c, d) {
        return jQuery.easing.easeOutQuad(x, t, b, c, d);
    },
    easeInOut: function(x, t, b, c, d) {
        return jQuery.easing.easeInOutQuad(x, t, b, c, d);
    },
    expoin: function(x, t, b, c, d) {
        return jQuery.easing.easeInExpo(x, t, b, c, d);
    },
    expoout: function(x, t, b, c, d) {
        return jQuery.easing.easeOutExpo(x, t, b, c, d);
    },
    expoinout: function(x, t, b, c, d) {
        return jQuery.easing.easeInOutExpo(x, t, b, c, d);
    },
    bouncein: function(x, t, b, c, d) {
        return jQuery.easing.easeInBounce(x, t, b, c, d);
    },
    bounceout: function(x, t, b, c, d) {
        return jQuery.easing.easeOutBounce(x, t, b, c, d);
    },
    bounceinout: function(x, t, b, c, d) {
        return jQuery.easing.easeInOutBounce(x, t, b, c, d);
    },
    elasin: function(x, t, b, c, d) {
        return jQuery.easing.easeInElastic(x, t, b, c, d);
    },
    elasout: function(x, t, b, c, d) {
        return jQuery.easing.easeOutElastic(x, t, b, c, d);
    },
    elasinout: function(x, t, b, c, d) {
        return jQuery.easing.easeInOutElastic(x, t, b, c, d);
    },
    backin: function(x, t, b, c, d) {
        return jQuery.easing.easeInBack(x, t, b, c, d);
    },
    backout: function(x, t, b, c, d) {
        return jQuery.easing.easeOutBack(x, t, b, c, d);
    },
    backinout: function(x, t, b, c, d) {
        return jQuery.easing.easeInOutBack(x, t, b, c, d);
    }
});
function colorFade(id, element, start, end, steps, speed, callback) {
    var startrgb, endrgb, er, eg, eb, step, rint, gint, bint, step;
    var target = document.getElementById(id);
    steps = steps || 20;
    speed = speed || 20;
    clearInterval(target.timer);
    endrgb = colorConv(end);
    er = endrgb[0];
    eg = endrgb[1];
    eb = endrgb[2];
    if (!target.r) {
        startrgb = colorConv(start);
        r = startrgb[0];
        g = startrgb[1];
        b = startrgb[2];
        target.r = r;
        target.g = g;
        target.b = b;
    }
    rint = Math.round(Math.abs(target.r - er) / steps);
    gint = Math.round(Math.abs(target.g - eg) / steps);
    bint = Math.round(Math.abs(target.b - eb) / steps);
    if (rint == 0) {
        rint = 1
    }
    if (gint == 0) {
        gint = 1
    }
    if (bint == 0) {
        bint = 1
    }
    target.step = 1;
    target.timer = setInterval(function() {
        animateColor(id, element, steps, er, eg, eb, rint, gint, bint, callback)
    }, speed);
}
function animateColor(id, element, steps, er, eg, eb, rint, gint, bint, callback) {
    var target = document.getElementById(id);
    var color;
    if (target.step <= steps) {
        var r = target.r;
        var g = target.g;
        var b = target.b;
        if (r >= er) {
            r = r - rint;
        } else {
            r = parseInt(r) + parseInt(rint);
        }
        if (g >= eg) {
            g = g - gint;
        } else {
            g = parseInt(g) + parseInt(gint);
        }
        if (b >= eb) {
            b = b - bint;
        } else {
            b = parseInt(b) + parseInt(bint);
        }
        color = 'rgb(' + r + ',' + g + ',' + b + ')';
        if (element == 'background') {
            target.style.backgroundColor = color;
        } else if (element == 'border') {
            target.style.borderColor = color;
        } else {
            target.style.color = color;
        }
        target.r = r;
        target.g = g;
        target.b = b;
        target.step = target.step + 1;
    } else {
        clearInterval(target.timer);
        color = 'rgb(' + er + ',' + eg + ',' + eb + ')';
        if (element == 'background') {
            target.style.backgroundColor = color;
        } else if (element == 'border') {
            target.style.borderColor = color;
        } else {
            target.style.color = color;
        }
        if (callback)
            callback();
    }
}
function colorConv(color) {
    var rgb = [parseInt(color.substring(0, 2), 16), parseInt(color.substring(2, 4), 16), parseInt(color.substring(4, 6), 16)];
    return rgb;
}
var qq = qq || {};
qq.extend = function(first, second) {
    for (var prop in second) {
        first[prop] = second[prop];
    }
};
qq.indexOf = function(arr, elt, from) {
    if (arr.indexOf)
        return arr.indexOf(elt, from);
    from = from || 0;
    var len = arr.length;
    if (from < 0)
        from += len;
    for (; from < len; from++) {
        if (from in arr && arr[from] === elt) {
            return from;
        }
    }
    return - 1;
};
qq.getUniqueId = (function() {
    var id = 0;
    return function() {
        return id++;
    };
})();
qq.attach = function(element, type, fn) {
    if (element.addEventListener) {
        element.addEventListener(type, fn, false);
    } else if (element.attachEvent) {
        element.attachEvent('on' + type, fn);
    }
};
qq.detach = function(element, type, fn) {
    if (element.removeEventListener) {
        element.removeEventListener(type, fn, false);
    } else if (element.attachEvent) {
        element.detachEvent('on' + type, fn);
    }
};
qq.preventDefault = function(e) {
    if (e.preventDefault) {
        e.preventDefault();
    } else {
        e.returnValue = false;
    }
};
qq.insertBefore = function(a, b) {
    b.parentNode.insertBefore(a, b);
};
qq.remove = function(element) {
    element.parentNode.removeChild(element);
};
qq.contains = function(parent, descendant) {
    if (parent == descendant)
        return true;
    if (parent.contains) {
        return parent.contains(descendant);
    } else {
        return !!(descendant.compareDocumentPosition(parent) & 8);
    }
};
qq.toElement = (function() {
    var div = document.createElement('div');
    return function(html) {
        div.innerHTML = html;
        var element = div.firstChild;
        div.removeChild(element);
        return element;
    };
})();
qq.css = function(element, styles) {
    if (styles.opacity != null) {
        if (typeof element.style.opacity != 'string' && typeof(element.filters) != 'undefined') {
            styles.filter = 'alpha(opacity=' + Math.round(100 * styles.opacity) + ')';
        }
    }
    qq.extend(element.style, styles);
};
qq.hasClass = function(element, name) {
    var re = new RegExp('(^| )' + name + '( |$)');
    return re.test(element.className);
};
qq.addClass = function(element, name) {
    if (!qq.hasClass(element, name)) {
        element.className += ' ' + name;
    }
};
qq.removeClass = function(element, name) {
    var re = new RegExp('(^| )' + name + '( |$)');
    element.className = element.className.replace(re, ' ').replace(/^\s+|\s+$/g, "");
};
qq.setText = function(element, text) {
    element.innerText = text;
    element.textContent = text;
};
qq.children = function(element) {
    var children = [], child = element.firstChild;
    while (child) {
        if (child.nodeType == 1) {
            children.push(child);
        }
        child = child.nextSibling;
    }
    return children;
};
qq.getByClass = function(element, className) {
    if (element.querySelectorAll) {
        return element.querySelectorAll('.' + className);
    }
    var result = [];
    var candidates = element.getElementsByTagName("*");
    var len = candidates.length;
    for (var i = 0; i < len; i++) {
        if (qq.hasClass(candidates[i], className)) {
            result.push(candidates[i]);
        }
    }
    return result;
};
qq.obj2url = function(obj, temp, prefixDone) {
    var uristrings = [], prefix = '&', add = function(nextObj, i) {
        var nextTemp = temp ? (/\[\]$/.test(temp)) ? temp: temp + '[' + i + ']': i;
        if ((nextTemp != 'undefined') && (i != 'undefined')) {
            uristrings.push((typeof nextObj === 'object') ? qq.obj2url(nextObj, nextTemp, true) : (Object.prototype.toString.call(nextObj) === '[object Function]') ? encodeURIComponent(nextTemp) + '=' + encodeURIComponent(nextObj()) : encodeURIComponent(nextTemp) + '=' + encodeURIComponent(nextObj));
        }
    };
    if (!prefixDone && temp) {
        prefix = (/\?/.test(temp)) ? (/\?$/.test(temp)) ? '' : '&' : '?';
        uristrings.push(temp);
        uristrings.push(qq.obj2url(obj));
    } else if ((Object.prototype.toString.call(obj) === '[object Array]') && (typeof obj != 'undefined')) {
        for (var i = 0, len = obj.length; i < len; ++i) {
            add(obj[i], i);
        }
    } else if ((typeof obj != 'undefined') && (obj !== null) && (typeof obj === "object")) {
        for (var i in obj) {
            add(obj[i], i);
        }
    } else {
        uristrings.push(encodeURIComponent(temp) + '=' + encodeURIComponent(obj));
    }
    return uristrings.join(prefix).replace(/^&/, '').replace(/%20/g, '+');
};
var qq = qq || {};
qq.FileUploaderBasic = function(o) {
    this._options = {
        debug: false,
        action: '/server/upload',
        params: {},
        button: null,
        multiple: false,
        maxConnections: 2,
        allowedExtensions: [],
        sizeLimit: 0,
        minSizeLimit: 0,
        onSubmit: function(id, fileName) {},
        onProgress: function(id, fileName, loaded, total) {},
        onComplete: function(id, fileName, responseJSON) {},
        onCancel: function(id, fileName) {},
        messages: {
            typeError: "Please upload a supported image. Only {extensions} are allowed.",
            sizeError: "{file} is too large, maximum file size is {sizeLimit}.",
            minSizeError: "{file} is too small, minimum file size is {minSizeLimit}.",
            emptyError: "{file} is empty, please select files again without it.",
            onLeave: "The files are being uploaded, if you leave now the upload will be cancelled."
        },
        showMessage: function(message) {
            alert(message);
        }
    };
    qq.extend(this._options, o);
    this._filesInProgress = 0;
    this._handler = this._createUploadHandler();
    if (this._options.button) {
        this._button = this._createUploadButton(this._options.button);
    }
    this._preventLeaveInProgress();
};
qq.FileUploaderBasic.prototype = {
    setParams: function(params) {
        this._options.params = params;
    },
    getInProgress: function() {
        return this._filesInProgress;
    },
    _createUploadButton: function(element) {
        var self = this;
        return new qq.UploadButton({
            element: element,
            multiple: this._options.multiple && qq.UploadHandlerXhr.isSupported(),
            onChange: function(input) {
                self._onInputChange(input);
            }
        });
    },
    _createUploadHandler: function() {
        var self = this, handlerClass;
        if (qq.UploadHandlerXhr.isSupported()) {
            handlerClass = 'UploadHandlerXhr';
        } else {
            handlerClass = 'UploadHandlerForm';
        }
        var handler = new qq[handlerClass]({
            debug: this._options.debug,
            action: this._options.action,
            maxConnections: this._options.maxConnections,
            onProgress: function(id, fileName, loaded, total) {
                self._onProgress(id, fileName, loaded, total);
                self._options.onProgress(id, fileName, loaded, total);
            },
            onComplete: function(id, fileName, result) {
                self._onComplete(id, fileName, result);
                self._options.onComplete(id, fileName, result);
            },
            onCancel: function(id, fileName) {
                self._onCancel(id, fileName);
                self._options.onCancel(id, fileName);
            }
        });
        return handler;
    },
    _preventLeaveInProgress: function() {
        var self = this;
        qq.attach(window, 'beforeunload', function(e) {
            if (!self._filesInProgress) {
                return;
            }
            var e = e || window.event;
            e.returnValue = self._options.messages.onLeave;
            return self._options.messages.onLeave;
        });
    },
    _onSubmit: function(id, fileName) {
        this._filesInProgress++;
    },
    _onProgress: function(id, fileName, loaded, total) {},
    _onComplete: function(id, fileName, result) {
        this._filesInProgress--;
        if (result.error) {
            this._options.showMessage(result.error);
        }
    },
    _onCancel: function(id, fileName) {
        this._filesInProgress--;
    },
    _onInputChange: function(input) {
        if (this._handler instanceof qq.UploadHandlerXhr) {
            this._uploadFileList(input.files);
        } else {
            if (this._validateFile(input)) {
                this._uploadFile(input);
            }
        }
        this._button.reset();
    },
    _uploadFileList: function(files) {
        for (var i = 0; i < files.length; i++) {
            if (!this._validateFile(files[i])) {
                return;
            }
        }
        for (var i = 0; i < files.length; i++) {
            this._uploadFile(files[i]);
        }
    },
    _uploadFile: function(fileContainer) {
        var id = this._handler.add(fileContainer);
        var fileName = this._handler.getName(id);
        if (this._options.onSubmit(id, fileName) !== false) {
            this._onSubmit(id, fileName);
            this._handler.upload(id, this._options.params);
        }
    },
    _validateFile: function(file) {
        var name, size;
        if (file.value) {
            name = file.value.replace(/.*(\/|\\)/, "");
        } else {
            name = file.fileName != null ? file.fileName : file.name;
            size = file.fileSize != null ? file.fileSize : file.size;
        }
        if (!this._isAllowedExtension(name)) {
            this._error('typeError', name);
            return false;
        } else if (size === 0) {
            this._error('emptyError', name);
            return false;
        } else if (size && this._options.sizeLimit && size > this._options.sizeLimit) {
            this._error('sizeError', name);
            return false;
        } else if (size && size < this._options.minSizeLimit) {
            this._error('minSizeError', name);
            return false;
        }
        return true;
    },
    _error: function(code, fileName) {
        var message = this._options.messages[code];
        function r(name, replacement) {
            message = message.replace(name, replacement);
        }
        r('{file}', this._formatFileName(fileName));
        r('{extensions}', this._options.allowedExtensions.join(', '));
        r('{sizeLimit}', this._formatSize(this._options.sizeLimit));
        r('{minSizeLimit}', this._formatSize(this._options.minSizeLimit));
        this._options.showMessage(message);
    },
    _formatFileName: function(name) {
        if (name.length > 33) {
            name = name.slice(0, 19) + '...' + name.slice( - 13);
        }
        return name;
    },
    _isAllowedExtension: function(fileName) {
        var ext = ( - 1 !== fileName.indexOf('.')) ? fileName.replace(/.*[.]/, '').toLowerCase(): '';
        var allowed = this._options.allowedExtensions;
        if (!allowed.length) {
            return true;
        }
        for (var i = 0; i < allowed.length; i++) {
            if (allowed[i].toLowerCase() == ext) {
                return true;
            }
        }
        return false;
    },
    _formatSize: function(bytes) {
        var i =- 1;
        do {
            bytes = bytes / 1024;
            i++;
        }
        while (bytes > 99);
        return Math.max(bytes, 0.1).toFixed(1) + ['kB', 'MB', 'GB', 'TB', 'PB', 'EB'][i];
    }
};
qq.FileUploader = function(o) {
    qq.FileUploaderBasic.apply(this, arguments);
    qq.extend(this._options, {
        element: null,
        listElement: null,
        template: '<div class="qq-uploader">' + '<div class="qq-upload-drop-area"><span>Drop files here to upload</span></div>' + '<div class="qq-upload-button">Upload Image</div>' + '<ul class="qq-upload-list"></ul>' + '</div>',
        fileTemplate: '<li>' + '<span class="qq-upload-file"></span>' + '<span class="qq-upload-spinner"></span>' + '<span class="qq-upload-size"></span>' + '<a class="qq-upload-cancel" href="#">Cancel</a>' + '<span class="qq-upload-failed-text">Failed</span>' + '</li>',
        classes: {
            button: 'qq-upload-button',
            drop: 'qq-upload-drop-area',
            dropActive: 'qq-upload-drop-area-active',
            list: 'qq-upload-list',
            file: 'qq-upload-file',
            spinner: 'qq-upload-spinner',
            size: 'qq-upload-size',
            cancel: 'qq-upload-cancel',
            success: 'qq-upload-success',
            fail: 'qq-upload-fail'
        }
    });
    qq.extend(this._options, o);
    this._element = this._options.element;
    this._element.innerHTML = this._options.template;
    this._listElement = this._options.listElement || this._find(this._element, 'list');
    this._classes = this._options.classes;
    this._button = this._createUploadButton(this._find(this._element, 'button'));
    this._bindCancelEvent();
    this._setupDragDrop();
};
qq.extend(qq.FileUploader.prototype, qq.FileUploaderBasic.prototype);
qq.extend(qq.FileUploader.prototype, {
    _find: function(parent, type) {
        var element = qq.getByClass(parent, this._options.classes[type])[0];
        if (!element) {
            throw new Error('element not found ' + type);
        }
        return element;
    },
    _setupDragDrop: function() {
        var self = this, dropArea = this._find(this._element, 'drop');
        var dz = new qq.UploadDropZone({
            element: dropArea,
            onEnter: function(e) {
                qq.addClass(dropArea, self._classes.dropActive);
                e.stopPropagation();
            },
            onLeave: function(e) {
                e.stopPropagation();
            },
            onLeaveNotDescendants: function(e) {
                qq.removeClass(dropArea, self._classes.dropActive);
            },
            onDrop: function(e) {
                dropArea.style.display = 'none';
                qq.removeClass(dropArea, self._classes.dropActive);
                self._uploadFileList(e.dataTransfer.files);
            }
        });
        dropArea.style.display = 'none';
        qq.attach(document, 'dragenter', function(e) {
            if (!dz._isValidFileDrag(e))
                return;
            dropArea.style.display = 'block';
        });
        qq.attach(document, 'dragleave', function(e) {
            if (!dz._isValidFileDrag(e))
                return;
            var relatedTarget = document.elementFromPoint(e.clientX, e.clientY);
            if (!relatedTarget || relatedTarget.nodeName == "HTML") {
                dropArea.style.display = 'none';
            }
        });
    },
    _onSubmit: function(id, fileName) {
        qq.FileUploaderBasic.prototype._onSubmit.apply(this, arguments);
        this._addToList(id, fileName);
    },
    _onProgress: function(id, fileName, loaded, total) {
        qq.FileUploaderBasic.prototype._onProgress.apply(this, arguments);
        var item = this._getItemByFileId(id);
        var size = this._find(item, 'size');
        size.style.display = 'inline';
        var text;
        if (loaded != total) {
            text = Math.round(loaded / total * 100) + '% from ' + this._formatSize(total);
        } else {
            text = this._formatSize(total);
        }
        qq.setText(size, text);
    },
    _onComplete: function(id, fileName, result) {
        qq.FileUploaderBasic.prototype._onComplete.apply(this, arguments);
        var item = this._getItemByFileId(id);
        qq.remove(this._find(item, 'cancel'));
        qq.remove(this._find(item, 'spinner'));
        if (result.success) {
            qq.addClass(item, this._classes.success);
        } else {
            qq.addClass(item, this._classes.fail);
        }
    },
    _addToList: function(id, fileName) {
        var item = qq.toElement(this._options.fileTemplate);
        item.qqFileId = id;
        var fileElement = this._find(item, 'file');
        qq.setText(fileElement, this._formatFileName(fileName));
        this._find(item, 'size').style.display = 'none';
        this._listElement.appendChild(item);
    },
    _getItemByFileId: function(id) {
        var item = this._listElement.firstChild;
        while (item) {
            if (item.qqFileId == id)
                return item;
            item = item.nextSibling;
        }
    },
    _bindCancelEvent: function() {
        var self = this, list = this._listElement;
        qq.attach(list, 'click', function(e) {
            e = e || window.event;
            var target = e.target || e.srcElement;
            if (qq.hasClass(target, self._classes.cancel)) {
                qq.preventDefault(e);
                var item = target.parentNode;
                self._handler.cancel(item.qqFileId);
                qq.remove(item);
            }
        });
    }
});
qq.UploadDropZone = function(o) {
    this._options = {
        element: null,
        onEnter: function(e) {},
        onLeave: function(e) {},
        onLeaveNotDescendants: function(e) {},
        onDrop: function(e) {}
    };
    qq.extend(this._options, o);
    this._element = this._options.element;
    this._disableDropOutside();
    this._attachEvents();
};
qq.UploadDropZone.prototype = {
    _disableDropOutside: function(e) {
        if (!qq.UploadDropZone.dropOutsideDisabled) {
            qq.attach(document, 'dragover', function(e) {
                if (e.dataTransfer) {
                    e.dataTransfer.dropEffect = 'none';
                    e.preventDefault();
                }
            });
            qq.UploadDropZone.dropOutsideDisabled = true;
        }
    },
    _attachEvents: function() {
        var self = this;
        qq.attach(self._element, 'dragover', function(e) {
            if (!self._isValidFileDrag(e))
                return;
            var effect = e.dataTransfer.effectAllowed;
            if (effect == 'move' || effect == 'linkMove') {
                e.dataTransfer.dropEffect = 'move';
            } else {
                e.dataTransfer.dropEffect = 'copy';
            }
            e.stopPropagation();
            e.preventDefault();
        });
        qq.attach(self._element, 'dragenter', function(e) {
            if (!self._isValidFileDrag(e))
                return;
            self._options.onEnter(e);
        });
        qq.attach(self._element, 'dragleave', function(e) {
            if (!self._isValidFileDrag(e))
                return;
            self._options.onLeave(e);
            var relatedTarget = document.elementFromPoint(e.clientX, e.clientY);
            if (qq.contains(this, relatedTarget))
                return;
            self._options.onLeaveNotDescendants(e);
        });
        qq.attach(self._element, 'drop', function(e) {
            if (!self._isValidFileDrag(e))
                return;
            e.preventDefault();
            self._options.onDrop(e);
        });
    },
    _isValidFileDrag: function(e) {
        var dt = e.dataTransfer, isWebkit = navigator.userAgent.indexOf("AppleWebKit")>-1;
        return dt && dt.effectAllowed != 'none' && (dt.files || (!isWebkit && dt.types.contains && dt.types.contains('Files')));
    }
};
qq.UploadButton = function(o) {
    this._options = {
        element: null,
        multiple: false,
        name: 'file',
        onChange: function(input) {},
        hoverClass: 'qq-upload-button-hover',
        focusClass: 'qq-upload-button-focus'
    };
    qq.extend(this._options, o);
    this._element = this._options.element;
    qq.css(this._element, {
        position: 'relative',
        overflow: 'hidden',
        direction: 'ltr'
    });
    this._input = this._createInput();
};
qq.UploadButton.prototype = {
    getInput: function() {
        return this._input;
    },
    reset: function() {
        if (this._input.parentNode) {
            qq.remove(this._input);
        }
        qq.removeClass(this._element, this._options.focusClass);
        this._input = this._createInput();
    },
    _createInput: function() {
        var input = document.createElement("input");
        if (this._options.multiple) {
            input.setAttribute("multiple", "multiple");
        } else
            input.setAttribute('accept', 'image/*');
        input.setAttribute("type", "file");
        input.setAttribute("name", this._options.name);
        qq.css(input, {
            position: 'absolute',
            right: 0,
            top: 0,
            fontFamily: 'Arial',
            fontSize: '118px',
            height: '32px',
            margin: 0,
            padding: 0,
            cursor: 'pointer',
            opacity: 0
        });
        this._element.appendChild(input);
        var self = this;
        qq.attach(input, 'change', function() {
            self._options.onChange(input);
        });
        qq.attach(input, 'mouseover', function() {
            qq.addClass(self._element, self._options.hoverClass);
        });
        qq.attach(input, 'mouseout', function() {
            qq.removeClass(self._element, self._options.hoverClass);
        });
        qq.attach(input, 'focus', function() {
            qq.addClass(self._element, self._options.focusClass);
        });
        qq.attach(input, 'blur', function() {
            qq.removeClass(self._element, self._options.focusClass);
        });
        if (window.attachEvent) {
            input.setAttribute('tabIndex', "-1");
        }
        return input;
    }
};
qq.UploadHandlerAbstract = function(o) {
    this._options = {
        debug: false,
        action: '/upload.php',
        maxConnections: 999,
        onProgress: function(id, fileName, loaded, total) {},
        onComplete: function(id, fileName, response) {},
        onCancel: function(id, fileName) {}
    };
    qq.extend(this._options, o);
    this._queue = [];
    this._params = [];
};
qq.UploadHandlerAbstract.prototype = {
    log: function(str) {
        if (this._options.debug && window.console)
            console.log('[uploader] ' + str);
    },
    add: function(file) {},
    upload: function(id, params) {
        var len = this._queue.push(id);
        var copy = {};
        qq.extend(copy, params);
        this._params[id] = copy;
        if (len <= this._options.maxConnections) {
            this._upload(id, this._params[id]);
        }
    },
    cancel: function(id) {
        this._cancel(id);
        this._dequeue(id);
    },
    cancelAll: function() {
        for (var i = 0; i < this._queue.length; i++) {
            this._cancel(this._queue[i]);
        }
        this._queue = [];
    },
    getName: function(id) {},
    getSize: function(id) {},
    getQueue: function() {
        return this._queue;
    },
    _upload: function(id) {},
    _cancel: function(id) {},
    _dequeue: function(id) {
        var i = qq.indexOf(this._queue, id);
        this._queue.splice(i, 1);
        var max = this._options.maxConnections;
        if (this._queue.length >= max && i < max) {
            var nextId = this._queue[max - 1];
            this._upload(nextId, this._params[nextId]);
        }
    }
};
qq.UploadHandlerForm = function(o) {
    qq.UploadHandlerAbstract.apply(this, arguments);
    this._inputs = {};
};
qq.extend(qq.UploadHandlerForm.prototype, qq.UploadHandlerAbstract.prototype);
qq.extend(qq.UploadHandlerForm.prototype, {
    add: function(fileInput) {
        fileInput.setAttribute('name', 'qqfile');
        var id = 'qq-upload-handler-iframe' + qq.getUniqueId();
        this._inputs[id] = fileInput;
        if (fileInput.parentNode) {
            qq.remove(fileInput);
        }
        return id;
    },
    getName: function(id) {
        return this._inputs[id].value.replace(/.*(\/|\\)/, "");
    },
    _cancel: function(id) {
        this._options.onCancel(id, this.getName(id));
        delete this._inputs[id];
        var iframe = document.getElementById(id);
        if (iframe) {
            iframe.setAttribute('src', 'javascript:false;');
            qq.remove(iframe);
        }
    },
    _upload: function(id, params) {
        var input = this._inputs[id];
        if (!input) {
            throw new Error('file with passed id was not added, or already uploaded or cancelled');
        }
        var fileName = this.getName(id);
        var iframe = this._createIframe(id);
        var form = this._createForm(iframe, params);
        form.appendChild(input);
        var self = this;
        this._attachLoadEvent(iframe, function() {
            self.log('iframe loaded');
            var response = self._getIframeContentJSON(iframe);
            self._options.onComplete(id, fileName, response);
            self._dequeue(id);
            delete self._inputs[id];
            setTimeout(function() {
                qq.remove(iframe);
            }, 1);
        });
        form.submit();
        qq.remove(form);
        return id;
    },
    _attachLoadEvent: function(iframe, callback) {
        qq.attach(iframe, 'load', function() {
            if (!iframe.parentNode) {
                return;
            }
            if (iframe.contentDocument && iframe.contentDocument.body && iframe.contentDocument.body.innerHTML == "false") {
                return;
            }
            callback();
        });
    },
    _getIframeContentJSON: function(iframe) {
        var doc = iframe.contentDocument ? iframe.contentDocument: iframe.contentWindow.document, response;
        this.log("converting iframe's innerHTML to JSON");
        this.log("innerHTML = " + doc.body.innerHTML);
        try {
            response = eval("(" + doc.body.innerHTML + ")");
        } catch (err) {
            response = {};
        }
        return response;
    },
    _createIframe: function(id) {
        var iframe = qq.toElement('<iframe src="javascript:false;" name="' + id + '" />');
        iframe.setAttribute('id', id);
        iframe.style.display = 'none';
        document.body.appendChild(iframe);
        return iframe;
    },
    _createForm: function(iframe, params) {
        var form = qq.toElement('<form method="post" enctype="multipart/form-data"></form>');
        var queryString = qq.obj2url(params, this._options.action);
        form.setAttribute('action', queryString);
        form.setAttribute('target', iframe.name);
        form.style.display = 'none';
        document.body.appendChild(form);
        return form;
    }
});
qq.UploadHandlerXhr = function(o) {
    qq.UploadHandlerAbstract.apply(this, arguments);
    this._files = [];
    this._xhrs = [];
    this._loaded = [];
};
qq.UploadHandlerXhr.isSupported = function() {
    var input = document.createElement('input');
    input.type = 'file';
    return ('multiple'in input && typeof File != "undefined" && typeof(new XMLHttpRequest()).upload != "undefined");
};
qq.extend(qq.UploadHandlerXhr.prototype, qq.UploadHandlerAbstract.prototype)
qq.extend(qq.UploadHandlerXhr.prototype, {
    add: function(file) {
        if (!(file instanceof File)) {
            throw new Error('Passed obj in not a File (in qq.UploadHandlerXhr)');
        }
        return this._files.push(file) - 1;
    },
    getName: function(id) {
        var file = this._files[id];
        return file.fileName != null ? file.fileName : file.name;
    },
    getSize: function(id) {
        var file = this._files[id];
        return file.fileSize != null ? file.fileSize : file.size;
    },
    getLoaded: function(id) {
        return this._loaded[id] || 0;
    },
    _upload: function(id, params) {
        var file = this._files[id], name = this.getName(id), size = this.getSize(id);
        this._loaded[id] = 0;
        var xhr = this._xhrs[id] = new XMLHttpRequest();
        var self = this;
        xhr.upload.onprogress = function(e) {
            if (e.lengthComputable) {
                self._loaded[id] = e.loaded;
                self._options.onProgress(id, name, e.loaded, e.total);
            }
        };
        xhr.onreadystatechange = function() {
            if (xhr.readyState == 4) {
                self._onComplete(id, xhr);
            }
        };
        params = params || {};
        params['qqfile'] = name;
        var queryString = qq.obj2url(params, this._options.action);
        xhr.open("POST", queryString, true);
        xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");
        xhr.setRequestHeader("X-File-Name", encodeURIComponent(name));
        xhr.setRequestHeader("Content-Type", "multipart/form-data");
        xhr.send(file);
    },
    _onComplete: function(id, xhr) {
        if (!this._files[id])
            return;
        var name = this.getName(id);
        var size = this.getSize(id);
        this._options.onProgress(id, name, size, size);
        if (xhr.status == 200) {
            this.log("xhr - server response received");
            this.log("responseText = " + xhr.responseText);
            var response;
            try {
                response = eval("(" + xhr.responseText + ")");
            } catch (err) {
                response = {};
            }
            this._options.onComplete(id, name, response);
        } else {
            this._options.onComplete(id, name, {});
        }
        this._files[id] = null;
        this._xhrs[id] = null;
        this._dequeue(id);
    },
    _cancel: function(id) {
        this._options.onCancel(id, this.getName(id));
        this._files[id] = null;
        if (this._xhrs[id]) {
            this._xhrs[id].abort();
            this._xhrs[id] = null;
        }
    }
});
(function(d) {
    function m() {
        var b = d("script:first"), a = b.css("color"), c = false;
        if (/^rgba/.test(a))
            c = true;
        else 
            try {
                c = a != b.css("color", "rgba(0, 0, 0, 0.5)").css("color");
                b.css("color", a)
        } catch (e) {}
        return c
    }
    function j(b, a, c) {
        var e = "rgb" + (d.support.rgba ? "a" : "") + "(" + parseInt(b[0] + c * (a[0] - b[0]), 10) + "," + parseInt(b[1] + c * (a[1] - b[1]), 10) + "," + parseInt(b[2] + c * (a[2] - b[2]), 10);
        if (d.support.rgba)
            e += "," + (b && a ? parseFloat(b[3] + c * (a[3] - b[3])) : 1);
        e += ")";
        return e
    }
    function g(b) {
        var a, c;
        if (a = /#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})/.exec(b))
            c = [parseInt(a[1], 16), parseInt(a[2], 16), parseInt(a[3], 16), 1];
        else if (a = /#([0-9a-fA-F])([0-9a-fA-F])([0-9a-fA-F])/.exec(b))
            c = [parseInt(a[1], 16) * 17, parseInt(a[2], 16) * 17, parseInt(a[3], 16) * 17, 1];
        else if (a = /rgb\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*\)/.exec(b))
            c = [parseInt(a[1]), parseInt(a[2]), parseInt(a[3]), 1];
        else if (a = /rgba\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9\.]*)\s*\)/.exec(b))
            c = [parseInt(a[1], 10), parseInt(a[2], 10), parseInt(a[3], 10), parseFloat(a[4])];
        return c
    }
    d.extend(true, d, {
        support: {
            rgba: m()
        }
    });
    var k = ["color", "backgroundColor", "borderBottomColor", "borderLeftColor", "borderRightColor", "borderTopColor", "outlineColor"];
    d.each(k, function(b, a) {
        d.Tween.propHooks[a] = {
            get: function(c) {
                return d(c.elem).css(a)
            },
            set: function(c) {
                var e = c.elem.style, i = g(d(c.elem).css(a)), h = g(c.end);
                c.run = function(f) {
                    e[a] = j(i, h, f)
                }
            }
        }
    });
    d.Tween.propHooks.borderColor = {
        set: function(b) {
            var a = b.elem.style, c = [], e = k.slice(2, 6);
            d.each(e, function(h, f) {
                c[f] = g(d(b.elem).css(f))
            });
            var i = g(b.end);
            b.run = function(h) {
                d.each(e, function(f, l) {
                    a[l] = j(c[l], i, h)
                })
            }
        }
    }
})(jQuery);
var dateFormat = function() {
    var token = /d{1,4}|m{1,4}|yy(?:yy)?|([HhMsTt])\1?|[LloSZ]|"[^"]*"|'[^']*'/g, timezone = /\b(?:[PMCEA][SDP]T|(?:Pacific|Mountain|Central|Eastern|Atlantic) (?:Standard|Daylight|Prevailing) Time|(?:GMT|UTC)(?:[-+]\d{4})?)\b/g, timezoneClip = /[^-+\dA-Z]/g, pad = function(val, len) {
        val = String(val);
        len = len || 2;
        while (val.length < len)
            val = "0" + val;
        return val;
    };
    return function(date, mask, utc) {
        var dF = dateFormat;
        if (arguments.length == 1 && Object.prototype.toString.call(date) == "[object String]"&&!/\d/.test(date)) {
            mask = date;
            date = undefined;
        }
        date = date ? new Date(date) : new Date;
        if (isNaN(date))
            throw SyntaxError("invalid date");
        mask = String(dF.masks[mask] || mask || dF.masks["default"]);
        if (mask.slice(0, 4) == "UTC:") {
            mask = mask.slice(4);
            utc = true;
        }
        var _ = utc ? "getUTC": "get", d = date[_ + "Date"](), D = date[_ + "Day"](), m = date[_ + "Month"](), y = date[_ + "FullYear"](), H = date[_ + "Hours"](), M = date[_ + "Minutes"](), s = date[_ + "Seconds"](), L = date[_ + "Milliseconds"](), o = utc ? 0: date.getTimezoneOffset(), flags = {
            d: d,
            dd: pad(d),
            ddd: dF.i18n.dayNames[D],
            dddd: dF.i18n.dayNames[D + 7],
            m: m + 1,
            mm: pad(m + 1),
            mmm: dF.i18n.monthNames[m],
            mmmm: dF.i18n.monthNames[m + 12],
            yy: String(y).slice(2),
            yyyy: y,
            h: H%12 || 12,
            hh: pad(H%12 || 12),
            H: H,
            HH: pad(H),
            M: M,
            MM: pad(M),
            s: s,
            ss: pad(s),
            l: pad(L, 3),
            L: pad(L > 99 ? Math.round(L / 10) : L),
            t: H < 12 ? "a": "p",
            tt: H < 12 ? "am": "pm",
            T: H < 12 ? "A": "P",
            TT: H < 12 ? "AM": "PM",
            Z: utc ? "UTC": (String(date).match(timezone) || [""]).pop().replace(timezoneClip, ""),
            o: (o > 0 ? "-" : "+") + pad(Math.floor(Math.abs(o) / 60) * 100 + Math.abs(o)%60, 4),
            S: ["th", "st", "nd", "rd"][d%10 > 3 ? 0: (d%100 - d%10 != 10) * d%10]
        };
        return mask.replace(token, function($0) {
            return $0 in flags ? flags[$0] : $0.slice(1, $0.length - 1);
        });
    };
}();
dateFormat.masks = {
    "default": "ddd mmm dd yyyy HH:MM:ss",
    shortDate: "m/d/yy",
    mediumDate: "mmm d, yyyy",
    longDate: "mmmm d, yyyy",
    fullDate: "dddd, mmmm d, yyyy",
    shortTime: "h:MM TT",
    mediumTime: "h:MM:ss TT",
    longTime: "h:MM:ss TT Z",
    isoDate: "yyyy-mm-dd",
    isoTime: "HH:MM:ss",
    isoDateTime: "yyyy-mm-dd'T'HH:MM:ss",
    isoUtcDateTime: "UTC:yyyy-mm-dd'T'HH:MM:ss'Z'"
};
dateFormat.i18n = {
    dayNames: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    monthNames: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
};
Date.prototype.format = function(mask, utc) {
    return dateFormat(this, mask, utc);
};
(function($) {
    jQuery.fn.extend({
        elastic: function() {
            var mimics = ['paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft', 'fontSize', 'lineHeight', 'fontFamily', 'width', 'fontWeight', 'border-top-width', 'border-right-width', 'border-bottom-width', 'border-left-width', 'borderTopStyle', 'borderTopColor', 'borderRightStyle', 'borderRightColor', 'borderBottomStyle', 'borderBottomColor', 'borderLeftStyle', 'borderLeftColor'];
            return this.each(function() {
                if (this.type !== 'textarea') {
                    return false;
                }
                var $textarea = jQuery(this), $twin = jQuery('<div />').css({
                    'position': 'absolute',
                    'display': 'none',
                    'word-wrap': 'break-word',
                    'white-space': 'pre-wrap'
                }), lineHeight = parseInt($textarea.css('line-height'), 10) || parseInt($textarea.css('font-size'), '10'), minheight = parseInt($textarea.css('height'), 10) || lineHeight * 3, maxheight = parseInt($textarea.css('max-height'), 10) || Number.MAX_VALUE, goalheight = 0;
                if (maxheight < 0) {
                    maxheight = Number.MAX_VALUE;
                }
                $twin.appendTo($textarea.parent());
                var i = mimics.length;
                while (i--) {
                    $twin.css(mimics[i].toString(), $textarea.css(mimics[i].toString()));
                }
                function setTwinWidth() {
                    var curatedWidth = Math.floor(parseInt($textarea.width(), 10));
                    if ($twin.width() !== curatedWidth) {
                        $twin.css({
                            'width': curatedWidth + 'px'
                        });
                        update(true);
                    }
                }
                function setHeightAndOverflow(height, overflow) {
                    var curratedHeight = Math.floor(parseInt(height, 10));
                    if ($textarea.height() !== curratedHeight) {
                        $textarea.css({
                            'height': curratedHeight + 'px',
                            'overflow': overflow
                        });
                    }
                }
                function update(forced) {
                    var textareaContent = $textarea.val().replace(/&/g, '&amp;').replace(/ {2}/g, '&nbsp;').replace(/<|>/g, '&gt;').replace(/\n/g, '<br />');
                    var twinContent = $twin.html().replace(/<br>/ig, '<br />');
                    if (forced || textareaContent + '&nbsp;' !== twinContent) {
                        $twin.html(textareaContent + '&nbsp;');
                        if (Math.abs($twin.height() + lineHeight - $textarea.height()) > 3) {
                            var goalheight = $twin.height();
                            if (goalheight >= maxheight) {
                                setHeightAndOverflow(maxheight, 'auto');
                            } else if (goalheight <= minheight) {
                                setHeightAndOverflow(minheight, 'hidden');
                            } else {
                                setHeightAndOverflow(goalheight, 'hidden');
                            }
                        }
                    }
                    if ($textarea.attr('id') == 'inputComment')
                        $container.isotope('reLayout');
                }
                $textarea.css({
                    'overflow': 'hidden'
                });
                $textarea.bind('keyup change cut paste', function() {
                    update();
                });
                $(window).bind('resize', setTwinWidth);
                $textarea.bind('resize', setTwinWidth);
                $textarea.bind('update', update);
                $textarea.bind('blur', function() {
                    if ($twin.height() < maxheight) {
                        if ($twin.height() > minheight) {
                            $textarea.height($twin.height());
                        } else {
                            $textarea.height(minheight);
                        }
                    }
                });
                $textarea.bind('input paste', function(e) {
                    setTimeout(update, 250);
                });
            });
        }
    });
})(jQuery); //#$.geocomplete()
(function(a, b, c, d) {
    function h(b, c) {
        this.options = a.extend(!0, {}, e, c), this.input = b, this.$input = a(b), this._defaults = e, this._name = "geocomplete", this.init()
    }
    var e = {
        bounds: !0,
        country: null,
        map: !1,
        details: !1,
        detailsAttribute: "name",
        location: !1,
        mapOptions: {
            zoom: 14,
            scrollwheel: !1,
            mapTypeId: "roadmap"
        },
        markerOptions: {
            draggable: !1
        },
        maxZoom: 16,
        types: ["geocode"]
    }, f = "street_address route intersection political country administrative_area_level_1 administrative_area_level_2 administrative_area_level_3 colloquial_area locality sublocality neighborhood premise subpremise postal_code natural_feature airport park point_of_interest post_box street_number floor room lat lng viewport location formatted_address location_type bounds".split(" "), g = "id url website vicinity reference name rating international_phone_number icon formatted_phone_number".split(" ");
    a.extend(h.prototype, {
        init: function() {
            this.initMap(), this.initMarker(), this.initGeocoder(), this.initDetails(), this.initLocation()
        },
        initMap: function() {
            if (!this.options.map)
                return;
            if (typeof this.options.map.setCenter == "function") {
                this.map = this.options.map;
                return 
            }
            this.map = new google.maps.Map(a(this.options.map)[0], this.options.mapOptions)
        },
        initMarker: function() {
            if (!this.map)
                return;
            var b = a.extend(this.options.markerOptions, {
                map: this.map
            });
            if (b.disabled)
                return;
            this.marker = new google.maps.Marker(b), google.maps.event.addListener(this.marker, "dragend", a.proxy(this.markerDragged, this))
        },
        initGeocoder: function() {
            var b = {
                types: this.options.types,
                bounds: this.options.bounds===!0 ? null: this.options.bounds,
                componentRestrictions: this.options.componentRestrictions
            };
            this.options.country && (b.componentRestrictions = {
                country: this.options.country
            }), this.autocomplete = new google.maps.places.Autocomplete(this.input, b), this.geocoder = new google.maps.Geocoder, this.map && this.options.bounds===!0 && this.autocomplete.bindTo("bounds", this.map), google.maps.event.addListener(this.autocomplete, "place_changed", a.proxy(this.placeChanged, this)), this.$input.keypress(function(a) {
                if (a.keyCode === 13)
                    return !1
            }), this.$input.bind("geocode", a.proxy(function() {
                this.find()
            }, this))
        },
        initDetails: function() {
            function e(a) {
                d[a] = b.find("[" + c + "=" + a + "]")
            }
            if (!this.options.details)
                return;
            var b = a(this.options.details), c = this.options.detailsAttribute, d = {};
            a.each(f, function(a, b) {
                e(b), e(b + "_short")
            }), a.each(g, function(a, b) {
                e(b)
            }), this.$details = b, this.details = d
        },
        initLocation: function() {
            var a = this.options.location, b;
            if (!a)
                return;
            if (typeof a == "string") {
                this.find(a);
                return 
            }
            a instanceof Array && (b = new google.maps.LatLng(a[0], a[1])), a instanceof google.maps.LatLng && (b = a), b && this.map && this.map.setCenter(b)
        },
        find: function(a) {
            this.geocode({
                address: a || this.$input.val()
            })
        },
        geocode: function(b) {
            this.options.bounds&&!b.bounds && (this.options.bounds===!0 ? b.bounds = this.map && this.map.getBounds() : b.bounds = this.options.bounds), this.options.country && (b.region = this.options.country), this.geocoder.geocode(b, a.proxy(this.handleGeocode, this))
        },
        handleGeocode: function(a, b) {
            if (b === google.maps.GeocoderStatus.OK) {
                var c = a[0];
                this.$input.val(c.formatted_address), this.update(c), a.length > 1 && this.trigger("geocode:multiple", a)
            } else 
                this.trigger("geocode:error", b)
        },
        trigger: function(a, b) {
            this.$input.trigger(a, [b])
        },
        center: function(a) {
            a.viewport ? (this.map.fitBounds(a.viewport), this.map.getZoom() > this.options.maxZoom && this.map.setZoom(this.options.maxZoom)) : (this.map.setZoom(this.options.maxZoom), this.map.setCenter(a.location)), this.marker && (this.marker.setPosition(a.location), this.marker.setAnimation(this.options.markerOptions.animation))
        },
        update: function(a) {
            this.map && this.center(a.geometry), this.$details && this.fillDetails(a), this.trigger("geocode:result", a)
        },
        fillDetails: function(b) {
            var c = {}, d = b.geometry, e = d.viewport, f = d.bounds;
            a.each(b.address_components, function(a, b) {
                var d = b.types[0];
                c[d] = b.long_name, c[d + "_short"] = b.short_name
            }), a.each(g, function(a, d) {
                c[d] = b[d]
            }), a.extend(c, {
                formatted_address: b.formatted_address,
                location_type: d.location_type || "PLACES",
                viewport: e,
                bounds: f,
                location: d.location,
                lat: d.location.lat(),
                lng: d.location.lng()
            }), a.each(this.details, a.proxy(function(a, b) {
                var d = c[a];
                this.setDetail(b, d)
            }, this)), this.data = c
        },
        setDetail: function(a, b) {
            b === d ? b = "" : typeof b.toUrlValue == "function" && (b = b.toUrlValue()), a.is(":input") ? a.val(b) : a.text(b)
        },
        markerDragged: function(a) {
            this.trigger("geocode:dragged", a.latLng)
        },
        resetMarker: function() {
            this.marker.setPosition(this.data.location), this.setDetail(this.details.lat, this.data.location.lat()), this.setDetail(this.details.lng, this.data.location.lng())
        },
        placeChanged: function() {
            var a = this.autocomplete.getPlace();
            a.geometry ? this.update(a) : this.find(a.name)
        }
    }), a.fn.geocomplete = function(b) {
        var c = "plugin_geocomplete";
        if (typeof b == "string") {
            var d = a(this).data(c) || a(this).geocomplete().data(c), e = d[b];
            return typeof e == "function" ? (e.apply(d, Array.prototype.slice.call(arguments, 1)), a(this)) : (arguments.length == 2 && (e = arguments[1]), e)
        }
        return this.each(function() {
            var d = a.data(this, c);
            d || (d = new h(this, b), a.data(this, c, d))
        })
    }
})(jQuery, window, document);
(function(window, $, undefined) {
    "use strict";
    $.infinitescroll = function infscr(options, callback, element) {
        this.element = $(element);
        if (!this._create(options, callback)) {
            this.failed = true;
        }
    };
    $.infinitescroll.defaults = {
        loading: {
            finished: undefined,
            finishedMsg: "<em>Congratulations, you've reached the end of the internet.</em>",
            img: "",
            msg: null,
            msgText: "<em>Loading the next set of posts...</em>",
            selector: null,
            speed: 'fast',
            start: undefined
        },
        state: {
            isDuringAjax: false,
            isInvalidPage: false,
            isDestroyed: false,
            isDone: false,
            isPaused: false,
            isBeyondMaxPage: false,
            currPage: 1
        },
        debug: false,
        behavior: undefined,
        binder: $(window),
        nextSelector: "div.navigation a:first",
        navSelector: "div.navigation",
        contentSelector: null,
        extraScrollPx: 150,
        itemSelector: "div.post",
        animate: false,
        pathParse: undefined,
        dataType: 'html',
        appendCallback: true,
        bufferPx: 40,
        errorCallback: function() {},
        infid: 0,
        pixelsFromNavToBottom: undefined,
        path: undefined,
        prefill: false,
        maxPage: undefined
    };
    $.infinitescroll.prototype = {
        _binding: function infscr_binding(binding) {
            var instance = this, opts = instance.options;
            opts.v = '2.0b2.120520';
            if (!!opts.behavior && this['_binding_' + opts.behavior] !== undefined) {
                this['_binding_' + opts.behavior].call(this);
                return;
            }
            if (binding !== 'bind' && binding !== 'unbind') {
                this._debug('Binding value  ' + binding + ' not valid');
                return false;
            }
            if (binding === 'unbind') {
                (this.options.binder).unbind('smartscroll.infscr.' + instance.options.infid);
            } else {
                (this.options.binder)[binding]('smartscroll.infscr.' + instance.options.infid, function() {
                    instance.scroll();
                });
            }
            this._debug('Binding', binding);
        },
        _create: function infscr_create(options, callback) {
            var opts = $.extend(true, {}, $.infinitescroll.defaults, options);
            this.options = opts;
            var $window = $(window);
            var instance = this;
            if (!instance._validate(options)) {
                return false;
            }
            var path = $(opts.nextSelector).attr('href');
            if (!path) {
                this._debug('Navigation selector not found');
                return false;
            }
            opts.path = opts.path || this._determinepath(path);
            opts.contentSelector = opts.contentSelector || this.element;
            opts.loading.selector = opts.loading.selector || opts.contentSelector;
            opts.loading.msg = opts.loading.msg || $('<div id="infscr-loading"><img alt="Loading..." src="' + opts.loading.img + '" /><div>' + opts.loading.msgText + '</div></div>');
            (new Image()).src = opts.loading.img;
            if (opts.pixelsFromNavToBottom === undefined) {
                opts.pixelsFromNavToBottom = $(document).height() - $(opts.navSelector).offset().top;
                this._debug("pixelsFromNavToBottom: " + opts.pixelsFromNavToBottom);
            }
            var self = this;
            opts.loading.start = opts.loading.start || function() {
                $(opts.navSelector).hide();
                opts.loading.msg.appendTo(opts.loading.selector).show(opts.loading.speed, $.proxy(function() {
                    this.beginAjax(opts);
                }, self));
            };
            opts.loading.finished = opts.loading.finished || function() {
                if (!opts.state.isBeyondMaxPage)
                    opts.loading.msg.fadeOut(opts.loading.speed);
            };
            opts.callback = function(instance, data, url) {
                if (!!opts.behavior && instance['_callback_' + opts.behavior] !== undefined) {
                    instance['_callback_' + opts.behavior].call($(opts.contentSelector)[0], data, url);
                }
                if (callback) {
                    callback.call($(opts.contentSelector)[0], data, opts, url);
                }
                if (opts.prefill) {
                    $window.bind("resize.infinite-scroll", instance._prefill);
                }
            };
            if (options.debug) {
                if (Function.prototype.bind && (typeof console === 'object' || typeof console === 'function') && typeof console.log === "object") {
                    ["log", "info", "warn", "error", "assert", "dir", "clear", "profile", "profileEnd"].forEach(function(method) {
                        console[method] = this.call(console[method], console);
                    }, Function.prototype.bind);
                }
            }
            this._setup();
            if (opts.prefill) {
                this._prefill();
            }
            return true;
        },
        _prefill: function infscr_prefill() {
            var instance = this;
            var $window = $(window);
            function needsPrefill() {
                return (instance.options.contentSelector.height() <= $window.height());
            }
            this._prefill = function() {
                if (needsPrefill()) {
                    instance.scroll();
                }
                $window.bind("resize.infinite-scroll", function() {
                    if (needsPrefill()) {
                        $window.unbind("resize.infinite-scroll");
                        instance.scroll();
                    }
                });
            };
            this._prefill();
        },
        _debug: function infscr_debug() {
            if (true !== this.options.debug) {
                return;
            }
            if (typeof console !== 'undefined' && typeof console.log === 'function') {
                if ((Array.prototype.slice.call(arguments)).length === 1 && typeof Array.prototype.slice.call(arguments)[0] === 'string') {
                    console.log((Array.prototype.slice.call(arguments)).toString());
                } else {
                    console.log(Array.prototype.slice.call(arguments));
                }
            } else if (!Function.prototype.bind && typeof console !== 'undefined' && typeof console.log === 'object') {
                Function.prototype.call.call(console.log, console, Array.prototype.slice.call(arguments));
            }
        },
        _determinepath: function infscr_determinepath(path) {
            var opts = this.options;
            if (!!opts.behavior && this['_determinepath_' + opts.behavior] !== undefined) {
                return this['_determinepath_' + opts.behavior].call(this, path);
            }
            if (!!opts.pathParse) {
                this._debug('pathParse manual');
                return opts.pathParse(path, this.options.state.currPage + 1);
            } else if (path.match(/^(.*?)\b2\b(.*?$)/)) {
                path = path.match(/^(.*?)\b2\b(.*?$)/).slice(1);
            } else if (path.match(/^(.*?)2(.*?$)/)) {
                if (path.match(/^(.*?page=)2(\/.*|$)/)) {
                    path = path.match(/^(.*?page=)2(\/.*|$)/).slice(1);
                    return path;
                }
                path = path.match(/^(.*?)2(.*?$)/).slice(1);
            } else {
                if (path.match(/^(.*?page=)1(\/.*|$)/)) {
                    path = path.match(/^(.*?page=)1(\/.*|$)/).slice(1);
                    return path;
                } else {
                    this._debug('Sorry, we couldn\'t parse your Next (Previous Posts) URL. Verify your the css selector points to the correct A tag. If you still get this error: yell, scream, and kindly ask for help at infinite-scroll.com.');
                    opts.state.isInvalidPage = true;
                }
            }
            this._debug('determinePath', path);
            return path;
        },
        _error: function infscr_error(xhr) {
            var opts = this.options;
            if (!!opts.behavior && this['_error_' + opts.behavior] !== undefined) {
                this['_error_' + opts.behavior].call(this, xhr);
                return;
            }
            if (xhr !== 'destroy' && xhr !== 'end') {
                xhr = 'unknown';
            }
            this._debug('Error', xhr);
            if (xhr === 'end' || opts.state.isBeyondMaxPage) {
                this._showdonemsg();
            }
            opts.state.isDone = true;
            opts.state.currPage = 1;
            opts.state.isPaused = false;
            opts.state.isBeyondMaxPage = false;
            this._binding('unbind');
        },
        _loadcallback: function infscr_loadcallback(box, data, url) {
            var opts = this.options, callback = this.options.callback, result = (opts.state.isDone) ? 'done': (!opts.appendCallback) ? 'no-append': 'append', frag;
            if (!!opts.behavior && this['_loadcallback_' + opts.behavior] !== undefined) {
                this['_loadcallback_' + opts.behavior].call(this, box, data);
                return;
            }
            switch (result) {
            case'done':
                this._showdonemsg();
                return false;
            case'no-append':
                if (opts.dataType === 'html') {
                    data = '<div>' + data + '</div>';
                    data = $(data).find(opts.itemSelector);
                }
                break;
            case'append':
                var children = box.children();
                if (children.length === 0) {
                    return this._error('end');
                }
                frag = document.createDocumentFragment();
                while (box[0].firstChild) {
                    frag.appendChild(box[0].firstChild);
                }
                this._debug('contentSelector', $(opts.contentSelector)[0]);
                $(opts.contentSelector)[0].appendChild(frag);
                data = children.get();
                break;
            }
            opts.loading.finished.call($(opts.contentSelector)[0], opts);
            if (opts.animate) {
                var scrollTo = $(window).scrollTop() + $(opts.loading.msg).height() + opts.extraScrollPx + 'px';
                $('html,body').animate({
                    scrollTop: scrollTo
                }, 800, function() {
                    opts.state.isDuringAjax = false;
                });
            }
            if (!opts.animate) {
                opts.state.isDuringAjax = false;
            }
            callback(this, data, url);
            if (opts.prefill) {
                this._prefill();
            }
        },
        _nearbottom: function infscr_nearbottom() {
            var opts = this.options, pixelsFromWindowBottomToBottom = 0 + $(document).height() - (opts.binder.scrollTop()) - $(window).height();
            if (!!opts.behavior && this['_nearbottom_' + opts.behavior] !== undefined) {
                return this['_nearbottom_' + opts.behavior].call(this);
            }
            this._debug('math:', pixelsFromWindowBottomToBottom, opts.pixelsFromNavToBottom, opts.bufferPx);
            return (pixelsFromWindowBottomToBottom - opts.bufferPx < opts.pixelsFromNavToBottom);
        },
        _pausing: function infscr_pausing(pause) {
            var opts = this.options;
            if (!!opts.behavior && this['_pausing_' + opts.behavior] !== undefined) {
                this['_pausing_' + opts.behavior].call(this, pause);
                return;
            }
            if (pause !== 'pause' && pause !== 'resume' && pause !== null) {
                this._debug('Invalid argument. Toggling pause value instead');
            }
            pause = (pause && (pause === 'pause' || pause === 'resume')) ? pause : 'toggle';
            switch (pause) {
            case'pause':
                opts.state.isPaused = true;
                break;
            case'resume':
                opts.state.isPaused = false;
                break;
            case'toggle':
                opts.state.isPaused=!opts.state.isPaused;
                break;
            }
            this._debug('Paused', opts.state.isPaused);
            return false;
        },
        _setup: function infscr_setup() {
            var opts = this.options;
            if (!!opts.behavior && this['_setup_' + opts.behavior] !== undefined) {
                this['_setup_' + opts.behavior].call(this);
                return;
            }
            this._binding('bind');
            return false;
        },
        _showdonemsg: function infscr_showdonemsg() {
            var opts = this.options;
            if (!!opts.behavior && this['_showdonemsg_' + opts.behavior] !== undefined) {
                this['_showdonemsg_' + opts.behavior].call(this);
                return;
            }
            opts.loading.msg.find('img').hide().parent().find('div').html(opts.loading.finishedMsg).animate({
                opacity: 1
            }, 2000, function() {
                $(this).parent().fadeOut(opts.loading.speed);
            });
            opts.errorCallback.call($(opts.contentSelector)[0], 'done');
        },
        _validate: function infscr_validate(opts) {
            for (var key in opts) {
                if (key.indexOf && key.indexOf('Selector')>-1 && $(opts[key]).length === 0) {
                    this._debug('Your ' + key + ' found no elements.');
                    return false;
                }
            }
            return true;
        },
        bind: function infscr_bind() {
            this._binding('bind');
        },
        destroy: function infscr_destroy() {
            this.options.state.isDestroyed = true;
            this.options.loading.finished();
            return this._error('destroy');
        },
        pause: function infscr_pause() {
            this._pausing('pause');
        },
        resume: function infscr_resume() {
            this._pausing('resume');
        },
        beginAjax: function infscr_ajax(opts) {
            var instance = this, path = opts.path, box, desturl, method, condition;
            opts.state.currPage++;
            if (opts.maxPage != undefined && opts.state.currPage > opts.maxPage) {
                opts.state.isBeyondMaxPage = true;
                this.destroy();
                return;
            }
            box = $(opts.contentSelector).is('table, tbody') ? $('<tbody/>') : $('<div/>');
            desturl = (typeof path === 'function') ? path(opts.state.currPage) : path.join(opts.state.currPage);
            instance._debug('heading into ajax', desturl);
            method = (opts.dataType === 'html' || opts.dataType === 'json') ? opts.dataType : 'html+callback';
            if (opts.appendCallback && opts.dataType === 'html') {
                method += '+callback';
            }
            switch (method) {
            case'html+callback':
                instance._debug('Using HTML via .load() method');
                box.load(desturl + ' ' + opts.itemSelector, undefined, function infscr_ajax_callback(responseText) {
                    instance._loadcallback(box, responseText, desturl);
                });
                break;
            case'html':
                instance._debug('Using ' + (method.toUpperCase()) + ' via $.ajax() method');
                $.ajax({
                    url: desturl,
                    dataType: opts.dataType,
                    complete: function infscr_ajax_callback(jqXHR, textStatus) {
                        condition = (typeof(jqXHR.isResolved) !== 'undefined') ? (jqXHR.isResolved()) : (textStatus === "success" || textStatus === "notmodified");
                        if (condition) {
                            instance._loadcallback(box, jqXHR.responseText, desturl);
                        } else {
                            instance._error('end');
                        }
                    }
                });
                break;
            case'json':
                instance._debug('Using ' + (method.toUpperCase()) + ' via $.ajax() method');
                $.ajax({
                    dataType: 'json',
                    type: 'GET',
                    url: desturl,
                    success: function(data, textStatus, jqXHR) {
                        condition = (typeof(jqXHR.isResolved) !== 'undefined') ? (jqXHR.isResolved()) : (textStatus === "success" || textStatus === "notmodified");
                        if (opts.appendCallback) {
                            if (opts.template !== undefined) {
                                var theData = opts.template(data);
                                box.append(theData);
                                if (condition) {
                                    instance._loadcallback(box, theData);
                                } else {
                                    instance._error('end');
                                }
                            } else {
                                instance._debug("template must be defined.");
                                instance._error('end');
                            }
                        } else {
                            if (condition) {
                                instance._loadcallback(box, data, desturl);
                            } else {
                                instance._error('end');
                            }
                        }
                    },
                    error: function() {
                        instance._debug("JSON ajax request failed.");
                        instance._error('end');
                    }
                });
                break;
            }
        },
        retrieve: function infscr_retrieve(pageNum) {
            pageNum = pageNum || null;
            var instance = this, opts = instance.options;
            if (!!opts.behavior && this['retrieve_' + opts.behavior] !== undefined) {
                this['retrieve_' + opts.behavior].call(this, pageNum);
                return;
            }
            if (opts.state.isDestroyed) {
                this._debug('Instance is destroyed');
                return false;
            }
            opts.state.isDuringAjax = true;
            opts.loading.start.call($(opts.contentSelector)[0], opts);
        },
        scroll: function infscr_scroll() {
            var opts = this.options, state = opts.state;
            if (!!opts.behavior && this['scroll_' + opts.behavior] !== undefined) {
                this['scroll_' + opts.behavior].call(this);
                return;
            }
            if (state.isDuringAjax || state.isInvalidPage || state.isDone || state.isDestroyed || state.isPaused) {
                return;
            }
            if (!this._nearbottom()) {
                return;
            }
            this.retrieve();
        },
        toggle: function infscr_toggle() {
            this._pausing();
        },
        unbind: function infscr_unbind() {
            this._binding('unbind');
        },
        update: function infscr_options(key) {
            if ($.isPlainObject(key)) {
                this.options = $.extend(true, this.options, key);
            }
        }
    };
    $.fn.infinitescroll = function infscr_init(options, callback) {
        var thisCall = typeof options;
        switch (thisCall) {
        case'string':
            var args = Array.prototype.slice.call(arguments, 1);
            this.each(function() {
                var instance = $.data(this, 'infinitescroll');
                if (!instance) {
                    return false;
                }
                if (!$.isFunction(instance[options]) || options.charAt(0) === "_") {
                    return false;
                }
                instance[options].apply(instance, args);
            });
            break;
        case'object':
            this.each(function() {
                var instance = $.data(this, 'infinitescroll');
                if (instance) {
                    instance.update(options);
                } else {
                    instance = new $.infinitescroll(options, callback, this);
                    if (!instance.failed) {
                        $.data(this, 'infinitescroll', instance);
                    }
                }
            });
            break;
        }
        return this;
    };
    var event = $.event, scrollTimeout;
    event.special.smartscroll = {
        setup: function() {
            $(this).bind("scroll", event.special.smartscroll.handler);
        },
        teardown: function() {
            $(this).unbind("scroll", event.special.smartscroll.handler);
        },
        handler: function(event, execAsap) {
            var context = this, args = arguments;
            event.type = "smartscroll";
            if (scrollTimeout) {
                clearTimeout(scrollTimeout);
            }
            scrollTimeout = setTimeout(function() {
                $(context).trigger('smartscroll', args);
            }, execAsap === "execAsap" ? 0 : 100);
        }
    };
    $.fn.smartscroll = function(fn) {
        return fn ? this.bind("smartscroll", fn) : this.trigger("smartscroll", ["execAsap"]);
    };
})(window, jQuery);
(function(a, b, c) {
    "use strict";
    var d = a.document, e = a.Modernizr, f = function(a) {
        return a.charAt(0).toUpperCase() + a.slice(1)
    }, g = "Moz Webkit O Ms".split(" "), h = function(a) {
        var b = d.documentElement.style, c;
        if (typeof b[a] == "string")
            return a;
        a = f(a);
        for (var e = 0, h = g.length; e < h; e++) {
            c = g[e] + a;
            if (typeof b[c] == "string")
                return c
        }
    }, i = h("transform"), j = h("transitionProperty"), k = {
        csstransforms: function() {
            return !!i
        },
        csstransforms3d: function() {
            var a=!!h("perspective");
            if (a) {
                var c = " -o- -moz- -ms- -webkit- -khtml- ".split(" "), d = "@media (" + c.join("transform-3d),(") + "modernizr)", e = b("<style>" + d + "{#modernizr{height:3px}}" + "</style>").appendTo("head"), f = b('<div id="modernizr" />').appendTo("html");
                a = f.height() === 3, f.remove(), e.remove()
            }
            return a
        },
        csstransitions: function() {
            return !!j
        }
    }, l;
    if (e)
        for (l in k)
            e.hasOwnProperty(l) || e.addTest(l, k[l]);
    else {
        e = a.Modernizr = {
            _version: "1.6ish: miniModernizr for Isotope"
        };
        var m = " ", n;
        for (l in k)
            n = k[l](), e[l] = n, m += " " + (n ? "" : "no-") + l;
        b("html").addClass(m)
    }
    if (e.csstransforms) {
        var o = e.csstransforms3d ? {
            translate: function(a) {
                return "translate(" + a[0] + "px, " + a[1] + "px) "
            },
            scale: function(a) {
                return "scale(" + a + ", " + a + ") "
            }
        }
        : {
            translate: function(a) {
                return "translate(" + a[0] + "px, " + a[1] + "px) "
            },
            scale: function(a) {
                return "scale(" + a + ") "
            }
        }, p = function(a, c, d) {
            var e = b.data(a, "isoTransform") || {}, f = {}, g, h = {}, j;
            f[c] = d, b.extend(e, f);
            for (g in e)
                j = e[g], h[g] = o[g](j);
            var k = h.translate || "", l = h.scale || "", m = k + l;
            b.data(a, "isoTransform", e), a.style[i] = m
        };
        b.cssNumber.scale=!0, b.cssHooks.scale = {
            set: function(a, b) {
                p(a, "scale", b)
            },
            get: function(a, c) {
                var d = b.data(a, "isoTransform");
                return d && d.scale ? d.scale : 1
            }
        }, b.fx.step.scale = function(a) {
            b.cssHooks.scale.set(a.elem, a.now + a.unit)
        }, b.cssNumber.translate=!0, b.cssHooks.translate = {
            set: function(a, b) {
                p(a, "translate", b)
            },
            get: function(a, c) {
                var d = b.data(a, "isoTransform");
                return d && d.translate ? d.translate : [0, 0]
            }
        }
    }
    var q, r;
    e.csstransitions && (q = {
        WebkitTransitionProperty: "webkitTransitionEnd",
        MozTransitionProperty: "transitionend",
        OTransitionProperty: "oTransitionEnd",
        transitionProperty: "transitionEnd"
    }
    [j], r = h("transitionDuration"));
    var s = b.event, t;
    s.special.smartresize = {
        setup: function() {
            b(this).bind("resize", s.special.smartresize.handler)
        },
        teardown: function() {
            b(this).unbind("resize", s.special.smartresize.handler)
        },
        handler: function(a, b) {
            var c = this, d = arguments;
            a.type = "smartresize", t && clearTimeout(t), t = setTimeout(function() {
                jQuery.event.handle.apply(c, d)
            }, b === "execAsap" ? 0 : 100)
        }
    }, b.fn.smartresize = function(a) {
        return a ? this.bind("smartresize", a) : this.trigger("smartresize", ["execAsap"])
    }, b.Isotope = function(a, c, d) {
        this.element = b(c), this._create(a), this._init(d)
    };
    var u = ["width", "height"], v = b(a);
    b.Isotope.settings = {
        resizable: !0,
        layoutMode: "masonry",
        containerClass: "isotope",
        itemClass: "isotope-item",
        hiddenClass: "isotope-hidden",
        hiddenStyle: {
            opacity: 0,
            scale: .001
        },
        visibleStyle: {
            opacity: 1,
            scale: 1
        },
        containerStyle: {
            position: "relative",
            overflow: "visible"
        },
        animationEngine: "best-available",
        animationOptions: {
            queue: !1,
            duration: 800
        },
        sortBy: "original-order",
        sortAscending: !0,
        resizesContainer: !0,
        transformsEnabled: !b.browser.opera,
        itemPositionDataEnabled: !1
    }, b.Isotope.prototype = {
        _create: function(a) {
            this.options = b.extend({}, b.Isotope.settings, a), this.styleQueue = [], this.elemCount = 0;
            var c = this.element[0].style;
            this.originalStyle = {};
            var d = u.slice(0);
            for (var e in this.options.containerStyle)
                d.push(e);
            for (var f = 0, g = d.length; f < g; f++)
                e = d[f], this.originalStyle[e] = c[e] || "";
            this.element.css(this.options.containerStyle), this._updateAnimationEngine(), this._updateUsingTransforms();
            var h = {
                "original-order": function(a, b) {
                    b.elemCount++;
                    return b.elemCount
                },
                random: function() {
                    return Math.random()
                }
            };
            this.options.getSortData = b.extend(this.options.getSortData, h), this.reloadItems(), this.offset = {
                left: parseInt(this.element.css("padding-left") || 0, 10),
                top: parseInt(this.element.css("padding-top") || 0, 10)
            };
            var i = this;
            setTimeout(function() {
                i.element.addClass(i.options.containerClass)
            }, 0), this.options.resizable && v.bind("smartresize.isotope", function() {
                i.resize()
            }), this.element.delegate("." + this.options.hiddenClass, "click", function() {
                return !1
            })
        },
        _getAtoms: function(a) {
            var b = this.options.itemSelector, c = b ? a.filter(b).add(a.find(b)): a, d = {
                position: "absolute"
            };
            this.usingTransforms && (d.left = 0, d.top = 0), c.css(d).addClass(this.options.itemClass), this.updateSortData(c, !0);
            return c
        },
        _init: function(a) {
            this.$filteredAtoms = this._filter(this.$allAtoms), this._sort(), this.reLayout(a)
        },
        option: function(a) {
            if (b.isPlainObject(a)) {
                this.options = b.extend(!0, this.options, a);
                var c;
                for (var d in a)
                    c = "_update" + f(d), this[c] && this[c]()
            }
        },
        _updateAnimationEngine: function() {
            var a = this.options.animationEngine.toLowerCase().replace(/[ _\-]/g, ""), b;
            switch (a) {
            case"css":
            case"none":
                b=!1;
                break;
            case"jquery":
                b=!0;
                break;
            default:
                b=!e.csstransitions
            }
            this.isUsingJQueryAnimation = b, this._updateUsingTransforms()
        },
        _updateTransformsEnabled: function() {
            this._updateUsingTransforms()
        },
        _updateUsingTransforms: function() {
            var a = this.usingTransforms = this.options.transformsEnabled && e.csstransforms && e.csstransitions&&!this.isUsingJQueryAnimation;
            a || (delete this.options.hiddenStyle.scale, delete this.options.visibleStyle.scale), this.getPositionStyles = a ? this._translate : this._positionAbs
        },
        _filter: function(a) {
            var b = this.options.filter === "" ? "*": this.options.filter;
            if (!b)
                return a;
            var c = this.options.hiddenClass, d = "." + c, e = a.filter(d), f = e;
            if (b !== "*") {
                f = e.filter(b);
                var g = a.not(d).not(b).addClass(c);
                this.styleQueue.push({
                    $el: g,
                    style: this.options.hiddenStyle
                })
            }
            this.styleQueue.push({
                $el: f,
                style: this.options.visibleStyle
            }), f.removeClass(c);
            return a.filter(b)
        },
        updateSortData: function(a, c) {
            var d = this, e = this.options.getSortData, f, g;
            a.each(function() {
                f = b(this), g = {};
                for (var a in e)
                    !c && a === "original-order" ? g[a] = b.data(this, "isotope-sort-data")[a] : g[a] = e[a](f, d);
                b.data(this, "isotope-sort-data", g)
            })
        },
        _sort: function() {
            var a = this.options.sortBy, b = this._getSorter, c = this.options.sortAscending ? 1: - 1, d = function(d, e) {
                var f = b(d, a), g = b(e, a);
                f === g && a !== "original-order" && (f = b(d, "original-order"), g = b(e, "original-order"));
                return (f > g ? 1 : f < g?-1 : 0) * c
            };
            this.$filteredAtoms.sort(d)
        },
        _getSorter: function(a, c) {
            return b.data(a, "isotope-sort-data")[c]
        },
        _translate: function(a, b) {
            return {
                translate: [a, b]
            }
        },
        _positionAbs: function(a, b) {
            return {
                left: a,
                top: b
            }
        },
        _pushPosition: function(a, b, c) {
            b = Math.round(b + this.offset.left), c = Math.round(c + this.offset.top);
            var d = this.getPositionStyles(b, c);
            this.styleQueue.push({
                $el: a,
                style: d
            }), this.options.itemPositionDataEnabled && a.data("isotope-item-position", {
                x: b,
                y: c
            })
        },
        layout: function(a, b) {
            var c = this.options.layoutMode;
            this["_" + c + "Layout"](a);
            if (this.options.resizesContainer) {
                var d = this["_" + c + "GetContainerSize"]();
                this.styleQueue.push({
                    $el: this.element,
                    style: d
                })
            }
            this._processStyleQueue(a, b), this.isLaidOut=!0
        },
        _processStyleQueue: function(a, c) {
            var d = this.isLaidOut ? this.isUsingJQueryAnimation ? "animate": "css": "css", f = this.options.animationOptions, g = this.options.onLayout, h, i, j, k;
            i = function(a, b) {
                b.$el[d](b.style, f)
            };
            if (this._isInserting && this.isUsingJQueryAnimation)
                i = function(a, b) {
                    h = b.$el.hasClass("no-transition") ? "css" : d, b.$el[h](b.style, f)
                };
            else if (c || g || f.complete) {
                var l=!1, m = [c, g, f.complete], n = this;
                j=!0, k = function() {
                    if (!l) {
                        var b;
                        for (var c = 0, d = m.length; c < d; c++)
                            b = m[c], typeof b == "function" && b.call(n.element, a, n);
                        l=!0
                    }
                };
                if (this.isUsingJQueryAnimation && d === "animate")
                    f.complete = k, j=!1;
                else if (e.csstransitions) {
                    var o = 0, p = this.styleQueue[0], s = p && p.$el, t;
                    while (!s ||!s.length) {
                        t = this.styleQueue[o++];
                        if (!t)
                            return;
                        s = t.$el
                    }
                    var u = parseFloat(getComputedStyle(s[0])[r]);
                    u > 0 && (i = function(a, b) {
                        b.$el[d](b.style, f).one(q, k)
                    }, j=!1)
                }
            }
            b.each(this.styleQueue, i), j && k(), this.styleQueue = []
        },
        resize: function() {
            this["_" + this.options.layoutMode + "ResizeChanged"]() && this.reLayout()
        },
        reLayout: function(a) {
            this["_" + this.options.layoutMode + "Reset"](), this.layout(this.$filteredAtoms, a)
        },
        addItems: function(a, b) {
            var c = this._getAtoms(a);
            this.$allAtoms = this.$allAtoms.add(c), b && b(c)
        },
        insert: function(a, b) {
            this.element.append(a);
            var c = this;
            this.addItems(a, function(a) {
                var d = c._filter(a);
                c._addHideAppended(d), c._sort(), c.reLayout(), c._revealAppended(d, b)
            })
        },
        appended: function(a, b) {
            var c = this;
            this.addItems(a, function(a) {
                c._addHideAppended(a), c.layout(a), c._revealAppended(a, b)
            })
        },
        _addHideAppended: function(a) {
            this.$filteredAtoms = this.$filteredAtoms.add(a), a.addClass("no-transition"), this._isInserting=!0, this.styleQueue.push({
                $el: a,
                style: this.options.hiddenStyle
            })
        },
        _revealAppended: function(a, b) {
            var c = this;
            setTimeout(function() {
                a.removeClass("no-transition"), c.styleQueue.push({
                    $el: a,
                    style: c.options.visibleStyle
                }), c._isInserting=!1, c._processStyleQueue(a, b)
            }, 10)
        },
        reloadItems: function() {
            this.$allAtoms = this._getAtoms(this.element.children())
        },
        remove: function(a, b) {
            var c = this, d = function() {
                c.$allAtoms = c.$allAtoms.not(a), a.remove(), b && b.call(this.element)
            };
            a.filter(":not(." + this.options.hiddenClass + ")").length ? (this.styleQueue.push({
                $el: a,
                style: this.options.hiddenStyle
            }), this.$filteredAtoms = this.$filteredAtoms.not(a), this._sort(), this.reLayout(d)) : d()
        },
        shuffle: function(a) {
            this.updateSortData(this.$allAtoms), this.options.sortBy = "random", this._sort(), this.reLayout(a)
        },
        destroy: function() {
            var a = this.usingTransforms, b = this.options;
            this.$allAtoms.removeClass(b.hiddenClass + " " + b.itemClass).each(function() {
                var b = this.style;
                b.position = "", b.top = "", b.left = "", b.opacity = "", a && (b[i] = "")
            });
            var c = this.element[0].style;
            for (var d in this.originalStyle)
                c[d] = this.originalStyle[d];
            this.element.unbind(".isotope").undelegate("." + b.hiddenClass, "click").removeClass(b.containerClass).removeData("isotope"), v.unbind(".isotope")
        },
        _getSegments: function(a) {
            var b = this.options.layoutMode, c = a ? "rowHeight": "columnWidth", d = a ? "height": "width", e = a ? "rows": "cols", g = this.element[d](), h, i = this.options[b] && this.options[b][c] || this.$filteredAtoms["outer" + f(d)](!0) || g;
            h = Math.floor(g / i), h = Math.max(h, 1), this[b][e] = h, this[b][c] = i
        },
        _checkIfSegmentsChanged: function(a) {
            var b = this.options.layoutMode, c = a ? "rows": "cols", d = this[b][c];
            this._getSegments(a);
            return this[b][c] !== d
        },
        _masonryReset: function() {
            this.masonry = {}, this._getSegments();
            var a = this.masonry.cols;
            this.masonry.colYs = [];
            while (a--)
                this.masonry.colYs.push(0)
        },
        _masonryLayout: function(a) {
            var c = this, d = c.masonry;
            a.each(function() {
                var a = b(this), e = Math.ceil(a.outerWidth(!0) / d.columnWidth);
                e = Math.min(e, d.cols);
                if (e === 1)
                    c._masonryPlaceBrick(a, d.colYs);
                else {
                    var f = d.cols + 1 - e, g = [], h, i;
                    for (i = 0; i < f; i++)
                        h = d.colYs.slice(i, i + e), g[i] = Math.max.apply(Math, h);
                    c._masonryPlaceBrick(a, g)
                }
            })
        },
        _masonryPlaceBrick: function(a, b) {
            var c = Math.min.apply(Math, b), d = 0;
            for (var e = 0, f = b.length; e < f; e++)
                if (b[e] === c) {
                    d = e;
                    break
                }
            var g = this.masonry.columnWidth * d, h = c;
            this._pushPosition(a, g, h);
            var i = c + a.outerHeight(!0), j = this.masonry.cols + 1 - f;
            for (e = 0; e < j; e++)
                this.masonry.colYs[d + e] = i
        },
        _masonryGetContainerSize: function() {
            var a = Math.max.apply(Math, this.masonry.colYs);
            return {
                height: a
            }
        },
        _masonryResizeChanged: function() {
            return this._checkIfSegmentsChanged()
        },
        _fitRowsReset: function() {
            this.fitRows = {
                x: 0,
                y: 0,
                height: 0
            }
        },
        _fitRowsLayout: function(a) {
            var c = this, d = this.element.width(), e = this.fitRows;
            a.each(function() {
                var a = b(this), f = a.outerWidth(!0), g = a.outerHeight(!0);
                e.x !== 0 && f + e.x > d && (e.x = 0, e.y = e.height), c._pushPosition(a, e.x, e.y), e.height = Math.max(e.y + g, e.height), e.x += f
            })
        },
        _fitRowsGetContainerSize: function() {
            return {
                height: this.fitRows.height
            }
        },
        _fitRowsResizeChanged: function() {
            return !0
        },
        _cellsByRowReset: function() {
            this.cellsByRow = {
                index: 0
            }, this._getSegments(), this._getSegments(!0)
        },
        _cellsByRowLayout: function(a) {
            var c = this, d = this.cellsByRow;
            a.each(function() {
                var a = b(this), e = d.index%d.cols, f = Math.floor(d.index / d.cols), g = (e + .5) * d.columnWidth - a.outerWidth(!0) / 2, h = (f + .5) * d.rowHeight - a.outerHeight(!0) / 2;
                c._pushPosition(a, g, h), d.index++
            })
        },
        _cellsByRowGetContainerSize: function() {
            return {
                height: Math.ceil(this.$filteredAtoms.length / this.cellsByRow.cols) * this.cellsByRow.rowHeight + this.offset.top
            }
        },
        _cellsByRowResizeChanged: function() {
            return this._checkIfSegmentsChanged()
        },
        _straightDownReset: function() {
            this.straightDown = {
                y: 0
            }
        },
        _straightDownLayout: function(a) {
            var c = this;
            a.each(function(a) {
                var d = b(this);
                c._pushPosition(d, 0, c.straightDown.y), c.straightDown.y += d.outerHeight(!0)
            })
        },
        _straightDownGetContainerSize: function() {
            return {
                height: this.straightDown.y
            }
        },
        _straightDownResizeChanged: function() {
            return !0
        },
        _masonryHorizontalReset: function() {
            this.masonryHorizontal = {}, this._getSegments(!0);
            var a = this.masonryHorizontal.rows;
            this.masonryHorizontal.rowXs = [];
            while (a--)
                this.masonryHorizontal.rowXs.push(0)
        },
        _masonryHorizontalLayout: function(a) {
            var c = this, d = c.masonryHorizontal;
            a.each(function() {
                var a = b(this), e = Math.ceil(a.outerHeight(!0) / d.rowHeight);
                e = Math.min(e, d.rows);
                if (e === 1)
                    c._masonryHorizontalPlaceBrick(a, d.rowXs);
                else {
                    var f = d.rows + 1 - e, g = [], h, i;
                    for (i = 0; i < f; i++)
                        h = d.rowXs.slice(i, i + e), g[i] = Math.max.apply(Math, h);
                    c._masonryHorizontalPlaceBrick(a, g)
                }
            })
        },
        _masonryHorizontalPlaceBrick: function(a, b) {
            var c = Math.min.apply(Math, b), d = 0;
            for (var e = 0, f = b.length; e < f; e++)
                if (b[e] === c) {
                    d = e;
                    break
                }
            var g = c, h = this.masonryHorizontal.rowHeight * d;
            this._pushPosition(a, g, h);
            var i = c + a.outerWidth(!0), j = this.masonryHorizontal.rows + 1 - f;
            for (e = 0; e < j; e++)
                this.masonryHorizontal.rowXs[d + e] = i
        },
        _masonryHorizontalGetContainerSize: function() {
            var a = Math.max.apply(Math, this.masonryHorizontal.rowXs);
            return {
                width: a
            }
        },
        _masonryHorizontalResizeChanged: function() {
            return this._checkIfSegmentsChanged(!0)
        },
        _fitColumnsReset: function() {
            this.fitColumns = {
                x: 0,
                y: 0,
                width: 0
            }
        },
        _fitColumnsLayout: function(a) {
            var c = this, d = this.element.height(), e = this.fitColumns;
            a.each(function() {
                var a = b(this), f = a.outerWidth(!0), g = a.outerHeight(!0);
                e.y !== 0 && g + e.y > d && (e.x = e.width, e.y = 0), c._pushPosition(a, e.x, e.y), e.width = Math.max(e.x + f, e.width), e.y += g
            })
        },
        _fitColumnsGetContainerSize: function() {
            return {
                width: this.fitColumns.width
            }
        },
        _fitColumnsResizeChanged: function() {
            return !0
        },
        _cellsByColumnReset: function() {
            this.cellsByColumn = {
                index: 0
            }, this._getSegments(), this._getSegments(!0)
        },
        _cellsByColumnLayout: function(a) {
            var c = this, d = this.cellsByColumn;
            a.each(function() {
                var a = b(this), e = Math.floor(d.index / d.rows), f = d.index%d.rows, g = (e + .5) * d.columnWidth - a.outerWidth(!0) / 2, h = (f + .5) * d.rowHeight - a.outerHeight(!0) / 2;
                c._pushPosition(a, g, h), d.index++
            })
        },
        _cellsByColumnGetContainerSize: function() {
            return {
                width: Math.ceil(this.$filteredAtoms.length / this.cellsByColumn.rows) * this.cellsByColumn.columnWidth
            }
        },
        _cellsByColumnResizeChanged: function() {
            return this._checkIfSegmentsChanged(!0)
        },
        _straightAcrossReset: function() {
            this.straightAcross = {
                x: 0
            }
        },
        _straightAcrossLayout: function(a) {
            var c = this;
            a.each(function(a) {
                var d = b(this);
                c._pushPosition(d, c.straightAcross.x, 0), c.straightAcross.x += d.outerWidth(!0)
            })
        },
        _straightAcrossGetContainerSize: function() {
            return {
                width: this.straightAcross.x
            }
        },
        _straightAcrossResizeChanged: function() {
            return !0
        }
    }, b.fn.imagesLoaded = function(a) {
        function i(a) {
            var c = a.target;
            c.src !== f && b.inArray(c, g)===-1 && (g.push(c), --e <= 0 && (setTimeout(h), d.unbind(".imagesLoaded", i)))
        }
        function h() {
            a.call(c, d)
        }
        var c = this, d = c.find("img").add(c.filter("img")), e = d.length, f = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==", g = [];
        e || h(), d.bind("load.imagesLoaded error.imagesLoaded", i).each(function() {
            var a = this.src;
            this.src = f, this.src = a
        });
        return c
    };
    var w = function(b) {
        a.console && a.console.error(b)
    };
    b.fn.isotope = function(a, c) {
        if (typeof a == "string") {
            var d = Array.prototype.slice.call(arguments, 1);
            this.each(function() {
                var c = b.data(this, "isotope");
                if (!c)
                    w("cannot call methods on isotope prior to initialization; attempted to call method '" + a + "'");
                else {
                    if (!b.isFunction(c[a]) || a.charAt(0) === "_") {
                        w("no such method '" + a + "' for isotope instance");
                        return 
                    }
                    c[a].apply(c, d)
                }
            })
        } else 
            this.each(function() {
                var d = b.data(this, "isotope");
                d ? (d.option(a), d._init(c)) : b.data(this, "isotope", new b.Isotope(a, this, c))
            });
        return this
    }
})(window, jQuery);
(function($) {
    var types = ['DOMMouseScroll', 'mousewheel'];
    if ($.event.fixHooks) {
        for (var i = types.length; i;) {
            $.event.fixHooks[types[--i]] = $.event.mouseHooks;
        }
    }
    $.event.special.mousewheel = {
        setup: function() {
            if (this.addEventListener) {
                for (var i = types.length; i;) {
                    this.addEventListener(types[--i], handler, false);
                }
            } else {
                this.onmousewheel = handler;
            }
        },
        teardown: function() {
            if (this.removeEventListener) {
                for (var i = types.length; i;) {
                    this.removeEventListener(types[--i], handler, false);
                }
            } else {
                this.onmousewheel = null;
            }
        }
    };
    $.fn.extend({
        mousewheel: function(fn) {
            return fn ? this.bind("mousewheel", fn) : this.trigger("mousewheel");
        },
        unmousewheel: function(fn) {
            return this.unbind("mousewheel", fn);
        }
    });
    function handler(event) {
        var orgEvent = event || window.event, args = [].slice.call(arguments, 1), delta = 0, returnValue = true, deltaX = 0, deltaY = 0;
        event = $.event.fix(orgEvent);
        event.type = "mousewheel";
        if (orgEvent.wheelDelta) {
            delta = orgEvent.wheelDelta / 120;
        }
        if (orgEvent.detail) {
            delta =- orgEvent.detail / 3;
        }
        deltaY = delta;
        if (orgEvent.axis !== undefined && orgEvent.axis === orgEvent.HORIZONTAL_AXIS) {
            deltaY = 0;
            deltaX =- 1 * delta;
        }
        if (orgEvent.wheelDeltaY !== undefined) {
            deltaY = orgEvent.wheelDeltaY / 120;
        }
        if (orgEvent.wheelDeltaX !== undefined) {
            deltaX =- 1 * orgEvent.wheelDeltaX / 120;
        }
        args.unshift(event, delta, deltaX, deltaY);
        return ($.event.dispatch || $.event.handle).apply(this, args);
    }
})(jQuery);
(function(b, a, c) {
    b.fn.jScrollPane = function(e) {
        function d(D, O) {
            var ay, Q = this, Y, aj, v, al, T, Z, y, q, az, aE, au, i, I, h, j, aa, U, ap, X, t, A, aq, af, am, G, l, at, ax, x, av, aH, f, L, ai = true, P = true, aG = false, k = false, ao = D.clone(false, false).empty(), ac = b.fn.mwheelIntent ? "mwheelIntent.jsp": "mousewheel.jsp";
            aH = D.css("paddingTop") + " " + D.css("paddingRight") + " " + D.css("paddingBottom") + " " + D.css("paddingLeft");
            f = (parseInt(D.css("paddingLeft"), 10) || 0) + (parseInt(D.css("paddingRight"), 10) || 0);
            function ar(aQ) {
                var aL, aN, aM, aJ, aI, aP, aO = false, aK = false;
                ay = aQ;
                if (Y === c) {
                    aI = D.scrollTop();
                    aP = D.scrollLeft();
                    D.css({
                        overflow: "hidden",
                        padding: 0
                    });
                    aj = D.innerWidth() + f;
                    v = D.innerHeight();
                    D.width(aj);
                    Y = b('<div class="jspPane newsOpen" />').css("padding", aH).append(D.children());
                    al = b('<div class="jspContainer newsOpen" />').css({
                        width: aj + "px",
                        height: v + "px"
                    }).append(Y).appendTo(D)
                } else {
                    D.css("width", "");
                    aO = ay.stickToBottom && K();
                    aK = ay.stickToRight && B();
                    aJ = D.innerWidth() + f != aj || D.outerHeight() != v;
                    if (aJ) {
                        aj = D.innerWidth() + f;
                        v = D.innerHeight();
                        al.css({
                            width: aj + "px",
                            height: v + "px"
                        })
                    }
                    if (!aJ && L == T && Y.outerHeight() == Z) {
                        D.width(aj);
                        return 
                    }
                    L = T;
                    Y.css("width", "");
                    D.width(aj);
                    al.find(">.jspVerticalBar,>.jspHorizontalBar").remove().end()
                }
                Y.css("overflow", "auto");
                if (aQ.contentWidth) {
                    T = aQ.contentWidth
                } else {
                    T = Y[0].scrollWidth
                }
                Z = Y[0].scrollHeight;
                Y.css("overflow", "");
                y = T / aj;
                q = Z / v;
                az = q > 1;
                aE = y > 1;
                if (!(aE || az)) {
                    D.removeClass("jspScrollable");
                    Y.css({
                        top: 0,
                        width: al.width() - f
                    });
                    n();
                    E();
                    R();
                    w()
                } else {
                    D.addClass("jspScrollable");
                    aL = ay.maintainPosition && (I || aa);
                    if (aL) {
                        aN = aC();
                        aM = aA()
                    }
                    aF();
                    z();
                    F();
                    if (aL) {
                        N(aK ? (T - aj) : aN, false);
                        M(aO ? (Z - v) : aM, false)
                    }
                    J();
                    ag();
                    an();
                    if (ay.enableKeyboardNavigation) {
                        S()
                    }
                    if (ay.clickOnTrack) {
                        p()
                    }
                    C();
                    if (ay.hijackInternalLinks) {
                        m()
                    }
                }
                if (ay.autoReinitialise&&!av) {
                    av = setInterval(function() {
                        ar(ay)
                    }, ay.autoReinitialiseDelay)
                } else {
                    if (!ay.autoReinitialise && av) {
                        clearInterval(av)
                    }
                }
                aI && D.scrollTop(0) && M(aI, false);
                aP && D.scrollLeft(0) && N(aP, false);
                D.trigger("jsp-initialised", [aE || az])
            }
            function aF() {
                if (az) {
                    al.append(b('<div class="jspVerticalBar newsOpen" />').append(b('<div class="jspCap jspCapTop newsOpen" />'), b('<div class="jspTrack newsOpen" />').append(b('<div class="jspDrag newsOpen" />').append(b('<div class="jspDragTop newsOpen" />'), b('<div class="jspDragBottom newsOpen" />'))), b('<div class="jspCap jspCapBottom newsOpen" />')));
                    U = al.find(">.jspVerticalBar");
                    ap = U.find(">.jspTrack");
                    au = ap.find(">.jspDrag");
                    if (ay.showArrows) {
                        aq = b('<a class="jspArrow jspArrowUp newsOpen" />').bind("mousedown.jsp", aD(0, - 1)).bind("click.jsp", aB);
                        af = b('<a class="jspArrow jspArrowDown newsOpen" />').bind("mousedown.jsp", aD(0, 1)).bind("click.jsp", aB);
                        if (ay.arrowScrollOnHover) {
                            aq.bind("mouseover.jsp", aD(0, - 1, aq));
                            af.bind("mouseover.jsp", aD(0, 1, af))
                        }
                        ak(ap, ay.verticalArrowPositions, aq, af)
                    }
                    t = v;
                    al.find(">.jspVerticalBar>.jspCap:visible,>.jspVerticalBar>.jspArrow").each(function() {
                        t -= b(this).outerHeight()
                    });
                    au.hover(function() {
                        au.addClass("jspHover")
                    }, function() {
                        au.removeClass("jspHover")
                    }).bind("mousedown.jsp", function(aI) {
                        b("html").bind("dragstart.jsp selectstart.jsp", aB);
                        au.addClass("jspActive");
                        var s = aI.pageY - au.position().top;
                        b("html").bind("mousemove.jsp", function(aJ) {
                            V(aJ.pageY - s, false)
                        }).bind("mouseup.jsp mouseleave.jsp", aw);
                        return false
                    });
                    o()
                }
            }
            function o() {
                ap.height(t + "px");
                I = 0;
                X = ay.verticalGutter + ap.outerWidth();
                Y.width(aj - X - f);
                try {
                    if (U.position().left === 0) {
                        Y.css("margin-left", X + "px")
                    }
                } catch (s) {}
            }
            function z() {
                if (aE) {
                    al.append(b('<div class="jspHorizontalBar newsOpen" />').append(b('<div class="jspCap jspCapLeft newsOpen" />'), b('<div class="jspTrack newsOpen" />').append(b('<div class="jspDrag newsOpen" />').append(b('<div class="jspDragLeft newsOpen" />'), b('<div class="jspDragRight newsOpen" />'))), b('<div class="jspCap jspCapRight newsOpen" />')));
                    am = al.find(">.jspHorizontalBar");
                    G = am.find(">.jspTrack");
                    h = G.find(">.jspDrag");
                    if (ay.showArrows) {
                        ax = b('<a class="jspArrow jspArrowLeft newsOpen" />').bind("mousedown.jsp", aD( - 1, 0)).bind("click.jsp", aB);
                        x = b('<a class="jspArrow jspArrowRight newsOpen" />').bind("mousedown.jsp", aD(1, 0)).bind("click.jsp", aB);
                        if (ay.arrowScrollOnHover) {
                            ax.bind("mouseover.jsp", aD( - 1, 0, ax));
                            x.bind("mouseover.jsp", aD(1, 0, x))
                        }
                        ak(G, ay.horizontalArrowPositions, ax, x)
                    }
                    h.hover(function() {
                        h.addClass("jspHover")
                    }, function() {
                        h.removeClass("jspHover")
                    }).bind("mousedown.jsp", function(aI) {
                        b("html").bind("dragstart.jsp selectstart.jsp", aB);
                        h.addClass("jspActive");
                        var s = aI.pageX - h.position().left;
                        b("html").bind("mousemove.jsp", function(aJ) {
                            W(aJ.pageX - s, false)
                        }).bind("mouseup.jsp mouseleave.jsp", aw);
                        return false
                    });
                    l = al.innerWidth();
                    ah()
                }
            }
            function ah() {
                al.find(">.jspHorizontalBar>.jspCap:visible,>.jspHorizontalBar>.jspArrow").each(function() {
                    l -= b(this).outerWidth()
                });
                G.width(l + "px");
                aa = 0
            }
            function F() {
                if (aE && az) {
                    var aI = G.outerHeight(), s = ap.outerWidth();
                    t -= aI;
                    b(am).find(">.jspCap:visible,>.jspArrow").each(function() {
                        l += b(this).outerWidth()
                    });
                    l -= s;
                    v -= s;
                    aj -= aI;
                    G.parent().append(b('<div class="jspCorner newsOpen" />').css("width", aI + "px"));
                    o();
                    ah()
                }
                if (aE) {
                    Y.width((al.outerWidth() - f) + "px")
                }
                Z = Y.outerHeight();
                q = Z / v;
                if (aE) {
                    at = Math.ceil(1 / y * l);
                    if (at > ay.horizontalDragMaxWidth) {
                        at = ay.horizontalDragMaxWidth
                    } else {
                        if (at < ay.horizontalDragMinWidth) {
                            at = ay.horizontalDragMinWidth
                        }
                    }
                    h.width(at + "px");
                    j = l - at;
                    ae(aa)
                }
                if (az) {
                    A = Math.ceil(1 / q * t);
                    if (A > ay.verticalDragMaxHeight) {
                        A = ay.verticalDragMaxHeight
                    } else {
                        if (A < ay.verticalDragMinHeight) {
                            A = ay.verticalDragMinHeight
                        }
                    }
                    au.height(A + "px");
                    i = t - A;
                    ad(I)
                }
            }
            function ak(aJ, aL, aI, s) {
                var aN = "before", aK = "after", aM;
                if (aL == "os") {
                    aL = /Mac/.test(navigator.platform) ? "after" : "split"
                }
                if (aL == aN) {
                    aK = aL
                } else {
                    if (aL == aK) {
                        aN = aL;
                        aM = aI;
                        aI = s;
                        s = aM
                    }
                }
                aJ[aN](aI)[aK](s)
            }
            function aD(aI, s, aJ) {
                return function() {
                    H(aI, s, this, aJ);
                    this.blur();
                    return false
                }
            }
            function H(aL, aK, aO, aN) {
                aO = b(aO).addClass("jspActive");
                var aM, aJ, aI = true, s = function() {
                    if (aL !== 0) {
                        Q.scrollByX(aL * ay.arrowButtonSpeed)
                    }
                    if (aK !== 0) {
                        Q.scrollByY(aK * ay.arrowButtonSpeed)
                    }
                    aJ = setTimeout(s, aI ? ay.initialDelay : ay.arrowRepeatFreq);
                    aI = false
                };
                s();
                aM = aN ? "mouseout.jsp" : "mouseup.jsp";
                aN = aN || b("html");
                aN.bind(aM, function() {
                    aO.removeClass("jspActive");
                    aJ && clearTimeout(aJ);
                    aJ = null;
                    aN.unbind(aM)
                })
            }
            function p() {
                w();
                if (az) {
                    ap.bind("mousedown.jsp", function(aN) {
                        if (aN.originalTarget === c || aN.originalTarget == aN.currentTarget) {
                            var aL = b(this), aO = aL.offset(), aM = aN.pageY - aO.top - I, aJ, aI = true, s = function() {
                                var aR = aL.offset(), aS = aN.pageY - aR.top - A / 2, aP = v * ay.scrollPagePercent, aQ = i * aP / (Z - v);
                                if (aM < 0) {
                                    if (I - aQ > aS) {
                                        Q.scrollByY( - aP)
                                    } else {
                                        V(aS)
                                    }
                                } else {
                                    if (aM > 0) {
                                        if (I + aQ < aS) {
                                            Q.scrollByY(aP)
                                        } else {
                                            V(aS)
                                        }
                                    } else {
                                        aK();
                                        return 
                                    }
                                }
                                aJ = setTimeout(s, aI ? ay.initialDelay : ay.trackClickRepeatFreq);
                                aI = false
                            }, aK = function() {
                                aJ && clearTimeout(aJ);
                                aJ = null;
                                b(document).unbind("mouseup.jsp", aK)
                            };
                            s();
                            b(document).bind("mouseup.jsp", aK);
                            return false
                        }
                    })
                }
                if (aE) {
                    G.bind("mousedown.jsp", function(aN) {
                        if (aN.originalTarget === c || aN.originalTarget == aN.currentTarget) {
                            var aL = b(this), aO = aL.offset(), aM = aN.pageX - aO.left - aa, aJ, aI = true, s = function() {
                                var aR = aL.offset(), aS = aN.pageX - aR.left - at / 2, aP = aj * ay.scrollPagePercent, aQ = j * aP / (T - aj);
                                if (aM < 0) {
                                    if (aa - aQ > aS) {
                                        Q.scrollByX( - aP)
                                    } else {
                                        W(aS)
                                    }
                                } else {
                                    if (aM > 0) {
                                        if (aa + aQ < aS) {
                                            Q.scrollByX(aP)
                                        } else {
                                            W(aS)
                                        }
                                    } else {
                                        aK();
                                        return 
                                    }
                                }
                                aJ = setTimeout(s, aI ? ay.initialDelay : ay.trackClickRepeatFreq);
                                aI = false
                            }, aK = function() {
                                aJ && clearTimeout(aJ);
                                aJ = null;
                                b(document).unbind("mouseup.jsp", aK)
                            };
                            s();
                            b(document).bind("mouseup.jsp", aK);
                            return false
                        }
                    })
                }
            }
            function w() {
                if (G) {
                    G.unbind("mousedown.jsp")
                }
                if (ap) {
                    ap.unbind("mousedown.jsp")
                }
            }
            function aw() {
                b("html").unbind("dragstart.jsp selectstart.jsp mousemove.jsp mouseup.jsp mouseleave.jsp");
                if (au) {
                    au.removeClass("jspActive")
                }
                if (h) {
                    h.removeClass("jspActive")
                }
            }
            function V(s, aI) {
                if (!az) {
                    return 
                }
                if (s < 0) {
                    s = 0
                } else {
                    if (s > i) {
                        s = i
                    }
                }
                if (aI === c) {
                    aI = ay.animateScroll
                }
                if (aI) {
                    Q.animate(au, "top", s, ad)
                } else {
                    au.css("top", s);
                    ad(s)
                }
            }
            function ad(aI) {
                if (aI === c) {
                    aI = au.position().top
                }
                al.scrollTop(0);
                I = aI;
                var aL = I === 0, aJ = I == i, aK = aI / i, s =- aK * (Z - v);
                if (ai != aL || aG != aJ) {
                    ai = aL;
                    aG = aJ;
                    D.trigger("jsp-arrow-change", [ai, aG, P, k])
                }
                u(aL, aJ);
                Y.css("top", s);
                D.trigger("jsp-scroll-y", [ - s, aL, aJ]).trigger("scroll")
            }
            function W(aI, s) {
                if (!aE) {
                    return 
                }
                if (aI < 0) {
                    aI = 0
                } else {
                    if (aI > j) {
                        aI = j
                    }
                }
                if (s === c) {
                    s = ay.animateScroll
                }
                if (s) {
                    Q.animate(h, "left", aI, ae)
                } else {
                    h.css("left", aI);
                    ae(aI)
                }
            }
            function ae(aI) {
                if (aI === c) {
                    aI = h.position().left
                }
                al.scrollTop(0);
                aa = aI;
                var aL = aa === 0, aK = aa == j, aJ = aI / j, s =- aJ * (T - aj);
                if (P != aL || k != aK) {
                    P = aL;
                    k = aK;
                    D.trigger("jsp-arrow-change", [ai, aG, P, k])
                }
                r(aL, aK);
                Y.css("left", s);
                D.trigger("jsp-scroll-x", [ - s, aL, aK]).trigger("scroll")
            }
            function u(aI, s) {
                if (ay.showArrows) {
                    aq[aI ? "addClass": "removeClass"]("jspDisabled");
                    af[s ? "addClass": "removeClass"]("jspDisabled")
                }
            }
            function r(aI, s) {
                if (ay.showArrows) {
                    ax[aI ? "addClass": "removeClass"]("jspDisabled");
                    x[s ? "addClass": "removeClass"]("jspDisabled")
                }
            }
            function M(s, aI) {
                var aJ = s / (Z - v);
                V(aJ * i, aI)
            }
            function N(aI, s) {
                var aJ = aI / (T - aj);
                W(aJ * j, s)
            }
            function ab(aV, aQ, aJ) {
                var aN, aK, aL, s = 0, aU = 0, aI, aP, aO, aS, aR, aT;
                try {
                    aN = b(aV)
                } catch (aM) {
                    return 
                }
                aK = aN.outerHeight();
                aL = aN.outerWidth();
                al.scrollTop(0);
                al.scrollLeft(0);
                while (!aN.is(".jspPane")) {
                    s += aN.position().top;
                    aU += aN.position().left;
                    aN = aN.offsetParent();
                    if (/^body|html$/i.test(aN[0].nodeName)) {
                        return 
                    }
                }
                aI = aA();
                aO = aI + v;
                if (s < aI || aQ) {
                    aR = s - ay.verticalGutter
                } else {
                    if (s + aK > aO) {
                        aR = s - v + aK + ay.verticalGutter
                    }
                }
                if (aR) {
                    M(aR, aJ)
                }
                aP = aC();
                aS = aP + aj;
                if (aU < aP || aQ) {
                    aT = aU - ay.horizontalGutter
                } else {
                    if (aU + aL > aS) {
                        aT = aU - aj + aL + ay.horizontalGutter
                    }
                }
                if (aT) {
                    N(aT, aJ)
                }
            }
            function aC() {
                return - Y.position().left
            }
            function aA() {
                return - Y.position().top
            }
            function K() {
                var s = Z - v;
                return (s > 20) && (s - aA() < 10)
            }
            function B() {
                var s = T - aj;
                return (s > 20) && (s - aC() < 10)
            }
            function ag() {
                al.unbind(ac).bind(ac, function(aL, aM, aK, aI) {
                    var aJ = aa, s = I;
                    Q.scrollBy(aK * ay.mouseWheelSpeed, - aI * ay.mouseWheelSpeed, false);
                    return aJ == aa && s == I
                })
            }
            function n() {
                al.unbind(ac)
            }
            function aB() {
                return false
            }
            function J() {
                Y.find(":input,a").unbind("focus.jsp").bind("focus.jsp", function(s) {
                    ab(s.target, false)
                })
            }
            function E() {
                Y.find(":input,a").unbind("focus.jsp")
            }
            function S() {
                var s, aI, aK = [];
                aE && aK.push(am[0]);
                az && aK.push(U[0]);
                Y.focus(function() {
                    D.focus()
                });
                D.attr("tabindex", 0).unbind("keydown.jsp keypress.jsp").bind("keydown.jsp", function(aN) {
                    if (aN.target !== this&&!(aK.length && b(aN.target).closest(aK).length)) {
                        return 
                    }
                    var aM = aa, aL = I;
                    switch (aN.keyCode) {
                    case 40:
                    case 38:
                    case 34:
                    case 32:
                    case 33:
                    case 39:
                    case 37:
                        s = aN.keyCode;
                        aJ();
                        break;
                    case 35:
                        M(Z - v);
                        s = null;
                        break;
                    case 36:
                        M(0);
                        s = null;
                        break
                    }
                    aI = aN.keyCode == s && aM != aa || aL != I;
                    return !aI
                }).bind("keypress.jsp", function(aL) {
                    if (aL.keyCode == s) {
                        aJ()
                    }
                    return !aI
                });
                if (ay.hideFocus) {
                    D.css("outline", "none");
                    if ("hideFocus"in al[0]) {
                        D.attr("hideFocus", true)
                    }
                } else {
                    D.css("outline", "");
                    if ("hideFocus"in al[0]) {
                        D.attr("hideFocus", false)
                    }
                }
                function aJ() {
                    var aM = aa, aL = I;
                    switch (s) {
                    case 40:
                        Q.scrollByY(ay.keyboardSpeed, false);
                        break;
                    case 38:
                        Q.scrollByY( - ay.keyboardSpeed, false);
                        break;
                    case 34:
                    case 32:
                        Q.scrollByY(v * ay.scrollPagePercent, false);
                        break;
                    case 33:
                        Q.scrollByY( - v * ay.scrollPagePercent, false);
                        break;
                    case 39:
                        Q.scrollByX(ay.keyboardSpeed, false);
                        break;
                    case 37:
                        Q.scrollByX( - ay.keyboardSpeed, false);
                        break
                    }
                    aI = aM != aa || aL != I;
                    return aI
                }
            }
            function R() {
                D.attr("tabindex", "-1").removeAttr("tabindex").unbind("keydown.jsp keypress.jsp")
            }
            function C() {
                if (location.hash && location.hash.length > 1) {
                    var aK, aI, aJ = escape(location.hash.substr(1));
                    try {
                        aK = b("#" + aJ + ', a[name="' + aJ + '"]')
                    } catch (s) {
                        return 
                    }
                    if (aK.length && Y.find(aJ)) {
                        if (al.scrollTop() === 0) {
                            aI = setInterval(function() {
                                if (al.scrollTop() > 0) {
                                    ab(aK, true);
                                    b(document).scrollTop(al.position().top);
                                    clearInterval(aI)
                                }
                            }, 50)
                        } else {
                            ab(aK, true);
                            b(document).scrollTop(al.position().top)
                        }
                    }
                }
            }
            function m() {
                if (b(document.body).data("jspHijack")) {
                    return 
                }
                b(document.body).data("jspHijack", true);
                b(document.body).delegate("a[href*=#]", "click", function(s) {
                    var aI = this.href.substr(0, this.href.indexOf("#")), aK = location.href, aO, aP, aJ, aM, aL, aN;
                    if (location.href.indexOf("#")!==-1) {
                        aK = location.href.substr(0, location.href.indexOf("#"))
                    }
                    if (aI !== aK) {
                        return 
                    }
                    aO = escape(this.href.substr(this.href.indexOf("#") + 1));
                    aP;
                    try {
                        aP = b("#" + aO + ', a[name="' + aO + '"]')
                    } catch (aQ) {
                        return 
                    }
                    if (!aP.length) {
                        return 
                    }
                    aJ = aP.closest(".jspScrollable");
                    aM = aJ.data("jsp");
                    aM.scrollToElement(aP, true);
                    if (aJ[0].scrollIntoView) {
                        aL = b(a).scrollTop();
                        aN = aP.offset().top;
                        if (aN < aL || aN > aL + b(a).height()) {
                            aJ[0].scrollIntoView()
                        }
                    }
                    s.preventDefault()
                })
            }
            function an() {
                var aJ, aI, aL, aK, aM, s = false;
                al.unbind("touchstart.jsp touchmove.jsp touchend.jsp click.jsp-touchclick").bind("touchstart.jsp", function(aN) {
                    var aO = aN.originalEvent.touches[0];
                    aJ = aC();
                    aI = aA();
                    aL = aO.pageX;
                    aK = aO.pageY;
                    aM = false;
                    s = true
                }).bind("touchmove.jsp", function(aQ) {
                    if (!s) {
                        return 
                    }
                    var aP = aQ.originalEvent.touches[0], aO = aa, aN = I;
                    Q.scrollTo(aJ + aL - aP.pageX, aI + aK - aP.pageY);
                    aM = aM || Math.abs(aL - aP.pageX) > 5 || Math.abs(aK - aP.pageY) > 5;
                    return aO == aa && aN == I
                }).bind("touchend.jsp", function(aN) {
                    s = false
                }).bind("click.jsp-touchclick", function(aN) {
                    if (aM) {
                        aM = false;
                        return false
                    }
                })
            }
            function g() {
                var s = aA(), aI = aC();
                D.removeClass("jspScrollable").unbind(".jsp");
                D.replaceWith(ao.append(Y.children()));
                ao.scrollTop(s);
                ao.scrollLeft(aI);
                if (av) {
                    clearInterval(av)
                }
            }
            b.extend(Q, {
                reinitialise: function(aI) {
                    aI = b.extend({}, ay, aI);
                    ar(aI)
                },
                scrollToElement: function(aJ, aI, s) {
                    ab(aJ, aI, s)
                },
                scrollTo: function(aJ, s, aI) {
                    N(aJ, aI);
                    M(s, aI)
                },
                scrollToX: function(aI, s) {
                    N(aI, s)
                },
                scrollToY: function(s, aI) {
                    M(s, aI)
                },
                scrollToPercentX: function(aI, s) {
                    N(aI * (T - aj), s)
                },
                scrollToPercentY: function(aI, s) {
                    M(aI * (Z - v), s)
                },
                scrollBy: function(aI, s, aJ) {
                    Q.scrollByX(aI, aJ);
                    Q.scrollByY(s, aJ)
                },
                scrollByX: function(s, aJ) {
                    var aI = aC() + Math[s < 0 ? "floor": "ceil"](s), aK = aI / (T - aj);
                    W(aK * j, aJ)
                },
                scrollByY: function(s, aJ) {
                    var aI = aA() + Math[s < 0 ? "floor": "ceil"](s), aK = aI / (Z - v);
                    V(aK * i, aJ)
                },
                positionDragX: function(s, aI) {
                    W(s, aI)
                },
                positionDragY: function(aI, s) {
                    V(aI, s)
                },
                animate: function(aI, aL, s, aK) {
                    var aJ = {};
                    aJ[aL] = s;
                    aI.animate(aJ, {
                        duration: ay.animateDuration,
                        easing: ay.animateEase,
                        queue: false,
                        step: aK
                    })
                },
                getContentPositionX: function() {
                    return aC()
                },
                getContentPositionY: function() {
                    return aA()
                },
                getContentWidth: function() {
                    return T
                },
                getContentHeight: function() {
                    return Z
                },
                getPercentScrolledX: function() {
                    return aC() / (T - aj)
                },
                getPercentScrolledY: function() {
                    return aA() / (Z - v)
                },
                getIsScrollableH: function() {
                    return aE
                },
                getIsScrollableV: function() {
                    return az
                },
                getContentPane: function() {
                    return Y
                },
                scrollToBottom: function(s) {
                    V(i, s)
                },
                hijackInternalLinks: b.noop,
                destroy: function() {
                    g()
                }
            });
            ar(O)
        }
        e = b.extend({}, b.fn.jScrollPane.defaults, e);
        b.each(["mouseWheelSpeed", "arrowButtonSpeed", "trackClickSpeed", "keyboardSpeed"], function() {
            e[this] = e[this] || e.speed
        });
        return this.each(function() {
            var f = b(this), g = f.data("jsp");
            if (g) {
                g.reinitialise(e)
            } else {
                b("script", f).filter('[type="text/javascript"],:not([type])').remove();
                g = new d(f, e);
                f.data("jsp", g)
            }
        })
    };
    b.fn.jScrollPane.defaults = {
        showArrows: false,
        maintainPosition: true,
        stickToBottom: false,
        stickToRight: false,
        clickOnTrack: true,
        autoReinitialise: false,
        autoReinitialiseDelay: 500,
        verticalDragMinHeight: 0,
        verticalDragMaxHeight: 99999,
        horizontalDragMinWidth: 0,
        horizontalDragMaxWidth: 99999,
        contentWidth: c,
        animateScroll: false,
        animateDuration: 300,
        animateEase: "linear",
        hijackInternalLinks: false,
        verticalGutter: 4,
        horizontalGutter: 4,
        mouseWheelSpeed: 0,
        arrowButtonSpeed: 0,
        arrowRepeatFreq: 50,
        arrowScrollOnHover: false,
        trackClickSpeed: 0,
        trackClickRepeatFreq: 70,
        verticalArrowPositions: "split",
        horizontalArrowPositions: "split",
        enableKeyboardNavigation: true,
        hideFocus: false,
        keyboardSpeed: 0,
        initialDelay: 300,
        speed: 30,
        scrollPagePercent: 0.8
    }
})(jQuery, this);
(function($) {
    var defaults = {
        topSpacing: 0,
        bottomSpacing: 0,
        className: 'is-sticky',
        wrapperClassName: 'sticky-wrapper',
        center: false,
        getWidthFrom: ''
    }, $window = $(window), $document = $(document), sticked = [], windowHeight = $window.height(), scroller = function() {
        var scrollTop = $window.scrollTop(), documentHeight = $document.height(), dwh = documentHeight - windowHeight, extra = (scrollTop > dwh) ? dwh - scrollTop: 0;
        for (var i = 0; i < sticked.length; i++) {
            var s = sticked[i], elementTop = s.stickyWrapper.offset().top, etse = elementTop - s.topSpacing - extra;
            if (scrollTop <= etse) {
                if (s.currentTop !== null) {
                    s.stickyElement.css('position', '').css('top', '');
                    s.stickyElement.parent().removeClass(s.className);
                    s.currentTop = null;
                }
            } else {
                var newTop = documentHeight - s.stickyElement.outerHeight()
                - s.topSpacing - s.bottomSpacing - scrollTop - extra;
                if (newTop < 0) {
                    newTop = newTop + s.topSpacing;
                } else {
                    newTop = s.topSpacing;
                }
                if (s.currentTop != newTop) {
                    s.stickyElement.css('position', 'fixed').css('top', newTop);
                    if (typeof s.getWidthFrom !== 'undefined') {
                        s.stickyElement.css('width', $(s.getWidthFrom).width());
                    }
                    s.stickyElement.parent().addClass(s.className);
                    s.currentTop = newTop;
                }
            }
        }
    }, resizer = function() {
        windowHeight = $window.height();
    }, methods = {
        init: function(options) {
            var o = $.extend(defaults, options);
            return this.each(function() {
                var stickyElement = $(this);
                var stickyId = stickyElement.attr('id');
                var wrapper = $('<div></div>').attr('id', stickyId + '-sticky-wrapper').addClass(o.wrapperClassName);
                stickyElement.wrapAll(wrapper);
                if (o.center) {
                    stickyElement.parent().css({
                        width: stickyElement.outerWidth(),
                        marginLeft: "auto",
                        marginRight: "auto"
                    });
                }
                if (stickyElement.css("float") == "right") {
                    stickyElement.css({
                        "float": "none"
                    }).parent().css({
                        "float": "right"
                    });
                }
                var stickyWrapper = stickyElement.parent();
                stickyWrapper.css('height', stickyElement.outerHeight());
                sticked.push({
                    topSpacing: o.topSpacing,
                    bottomSpacing: o.bottomSpacing,
                    stickyElement: stickyElement,
                    currentTop: null,
                    stickyWrapper: stickyWrapper,
                    className: o.className,
                    getWidthFrom: o.getWidthFrom
                });
            });
        },
        update: scroller
    };
    if (window.addEventListener) {
        window.addEventListener('scroll', scroller, false);
        window.addEventListener('resize', resizer, false);
    } else if (window.attachEvent) {
        window.attachEvent('onscroll', scroller);
        window.attachEvent('onresize', resizer);
    }
    $.fn.sticky = function(method) {
        if (methods[method]) {
            return methods[method].apply(this, Array.prototype.slice.call(arguments, 1));
        } else if (typeof method === 'object' ||!method) {
            return methods.init.apply(this, arguments);
        } else {
            $.error('Method ' + method + ' does not exist on jQuery.sticky');
        }
    };
    $(function() {
        setTimeout(scroller, 0);
    });
})(jQuery);
(function($) {
    function maybeCall(thing, ctx) {
        return (typeof thing == 'function') ? (thing.call(ctx)) : thing;
    };
    function Tipsy(element, options) {
        this.$element = $(element);
        this.options = options;
        this.enabled = true;
        this.fixTitle();
    };
    Tipsy.prototype = {
        show: function() {
            var title = this.getTitle();
            if (title && this.enabled) {
                var $tip = this.tip();
                $tip.find('.tipsy-inner')[this.options.html ? 'html': 'text'](title);
                $tip[0].className = 'tipsy';
                $tip.remove().css({
                    top: 0,
                    left: 0,
                    visibility: 'hidden',
                    display: 'block'
                }).prependTo(document.body);
                var pos = $.extend({}, this.$element.offset(), {
                    width: this.$element[0].offsetWidth,
                    height: this.$element[0].offsetHeight
                });
                var actualWidth = $tip[0].offsetWidth, actualHeight = $tip[0].offsetHeight, gravity = maybeCall(this.options.gravity, this.$element[0]);
                var tp;
                switch (gravity.charAt(0)) {
                case'n':
                    tp = {
                        top: pos.top + pos.height + this.options.offset,
                        left: pos.left + pos.width / 2 - actualWidth / 2
                    };
                    break;
                case's':
                    tp = {
                        top: pos.top - actualHeight - this.options.offset,
                        left: pos.left + pos.width / 2 - actualWidth / 2
                    };
                    break;
                case'e':
                    tp = {
                        top: pos.top + pos.height / 2 - actualHeight / 2,
                        left: pos.left - actualWidth - this.options.offset
                    };
                    break;
                case'w':
                    tp = {
                        top: pos.top + pos.height / 2 - actualHeight / 2,
                        left: pos.left + pos.width + this.options.offset
                    };
                    break;
                }
                if (gravity.length == 2) {
                    if (gravity.charAt(1) == 'w') {
                        tp.left = pos.left + pos.width / 2 - 15;
                    } else {
                        tp.left = pos.left + pos.width / 2 - actualWidth + 15;
                    }
                }
                $tip.css(tp).addClass('tipsy-' + gravity);
                $tip.find('.tipsy-arrow')[0].className = 'tipsy-arrow tipsy-arrow-' + gravity.charAt(0);
                if (this.options.className) {
                    $tip.addClass(maybeCall(this.options.className, this.$element[0]));
                }
                if (this.options.fade) {
                    $tip.stop(true, true).css({
                        opacity: 0,
                        display: 'block',
                        visibility: 'visible'
                    }).animate({
                        opacity: this.options.opacity
                    });
                } else {
                    $tip.css({
                        visibility: 'visible',
                        opacity: this.options.opacity
                    });
                }
            }
        },
        hide: function() {
            this.tip().remove();
        },
        fixTitle: function() {
            var $e = this.$element;
            if ($e.attr('title') || typeof($e.attr('original-title')) != 'string') {
                $e.attr('original-title', $e.attr('title') || '').removeAttr('title');
            }
        },
        getTitle: function() {
            var title, $e = this.$element, o = this.options;
            this.fixTitle();
            var title, o = this.options;
            if (typeof o.title == 'string') {
                title = $e.attr(o.title == 'title' ? 'original-title' : o.title);
            } else if (typeof o.title == 'function') {
                title = o.title.call($e[0]);
            }
            title = ('' + title).replace(/(^\s*|\s*$)/, "");
            return title || o.fallback;
        },
        tip: function() {
            if (!this.$tip) {
                this.$tip = $('<div class="tipsy"></div>').html('<div class="tipsy-arrow"></div><div class="tipsy-inner"></div>');
            }
            return this.$tip;
        },
        validate: function() {
            if (!this.$element[0].parentNode) {
                this.hide();
                this.$element = null;
                this.options = null;
            }
        },
        enable: function() {
            this.enabled = true;
        },
        disable: function() {
            this.enabled = false;
        },
        toggleEnabled: function() {
            this.enabled=!this.enabled;
        }
    };
    $.fn.tipsy = function(options) {
        if (options === true) {
            return this.data('tipsy');
        } else if (typeof options == 'string') {
            var tipsy = this.data('tipsy');
            if (tipsy)
                tipsy[options]();
            return this;
        }
        options = $.extend({}, $.fn.tipsy.defaults, options);
        function get(ele) {
            var tipsy = $.data(ele, 'tipsy');
            if (!tipsy) {
                tipsy = new Tipsy(ele, $.fn.tipsy.elementOptions(ele, options));
                $.data(ele, 'tipsy', tipsy);
            }
            return tipsy;
        }
        function enter() {
            var tipsy = get(this);
            tipsy.hoverState = 'in';
            if (options.delayIn == 0) {
                tipsy.show();
            } else {
                tipsy.fixTitle();
                setTimeout(function() {
                    if (tipsy.hoverState == 'in')
                        tipsy.show();
                }, options.delayIn);
            }
        };
        function leave() {
            var tipsy = get(this);
            tipsy.hoverState = 'out';
            if (options.delayOut == 0) {
                tipsy.hide();
            } else {
                setTimeout(function() {
                    if (tipsy.hoverState == 'out')
                        tipsy.hide();
                }, options.delayOut);
            }
        };
        if (!options.live)
            this.each(function() {
                get(this);
            });
        if (options.trigger != 'manual') {
            var binder = options.live ? 'live': 'bind', eventIn = options.trigger == 'hover' ? 'mouseenter': 'focus', eventOut = options.trigger == 'hover' ? 'mouseleave': 'blur';
            this[binder](eventIn, enter)[binder](eventOut, leave);
        }
        return this;
    };
    $.fn.tipsy.defaults = {
        className: null,
        delayIn: 0,
        delayOut: 0,
        fade: false,
        fallback: '',
        gravity: 'n',
        html: false,
        live: false,
        offset: 0,
        opacity: 0.8,
        title: 'title',
        trigger: 'hover'
    };
    $.fn.tipsy.elementOptions = function(ele, options) {
        return $.metadata ? $.extend({}, options, $(ele).metadata()) : options;
    };
    $.fn.tipsy.autoNS = function() {
        return $(this).offset().top > ($(document).scrollTop() + $(window).height() / 2) ? 's' : 'n';
    };
    $.fn.tipsy.autoWE = function() {
        return $(this).offset().left > ($(document).scrollLeft() + $(window).width() / 2) ? 'e' : 'w';
    };
    $.fn.tipsy.autoBounds = function(margin, prefer) {
        return function() {
            var dir = {
                ns: prefer[0],
                ew: (prefer.length > 1 ? prefer[1] : false)
            }, boundTop = $(document).scrollTop() + margin, boundLeft = $(document).scrollLeft() + margin, $this = $(this);
            if ($this.offset().top < boundTop)
                dir.ns = 'n';
            if ($this.offset().left < boundLeft)
                dir.ew = 'w';
            if ($(window).width() + $(document).scrollLeft() - $this.offset().left < margin)
                dir.ew = 'e';
            if ($(window).height() + $(document).scrollTop() - $this.offset().top < margin)
                dir.ns = 's';
            return dir.ns + (dir.ew ? dir.ew : '');
        }
    };
})(jQuery);
(function($) {
    $.fn.UItoTop = function(options) {
        var defaults = {
            text: 'To Top',
            min: 200,
            inDelay: 600,
            outDelay: 400,
            containerID: 'toTop',
            containerHoverID: 'toTopHover',
            scrollSpeed: 1200,
            easingType: 'linear'
        }, settings = $.extend(defaults, options), containerIDhash = '#' + settings.containerID, containerHoverIDHash = '#' + settings.containerHoverID;
        $('body').append('<a href="#" id="' + settings.containerID + '">' + settings.text + '</a>');
        $(containerIDhash).hide().on('click.UItoTop', function() {
            $('html, body').stop().animate({
                scrollTop: 0
            }, settings.scrollSpeed, settings.easingType);
            $('#' + settings.containerHoverID, this).stop().animate({
                'opacity': 0
            }, settings.inDelay, settings.easingType);
            return false;
        }).prepend('<span id="' + settings.containerHoverID + '"></span>').hover(function() {
            $(containerHoverIDHash, this).stop().animate({
                'opacity': 1
            }, 600, 'linear');
        }, function() {
            $(containerHoverIDHash, this).stop().animate({
                'opacity': 0
            }, 700, 'linear');
        });
        $(window).scroll(function() {
            var sd = $(window).scrollTop();
            if (typeof document.body.style.maxHeight === "undefined") {
                $(containerIDhash).css({
                    'position': 'absolute',
                    'top': sd + $(window).height() - 50
                });
            }
            if (sd > settings.min)
                $(containerIDhash).fadeIn(settings.inDelay);
            else
                $(containerIDhash).fadeOut(settings.Outdelay);
        });
    };
})(jQuery);
(function(n, t, i) {
    var g = "TEXTAREA", d = "function", nt = "password", c = "maxLength", v = "type", r = "", u=!0, rt = "placeholder", h=!1, tt = "watermark", s = tt, o = "watermarkClass", w = "watermarkFocus", a = "watermarkSubmit", b = "watermarkMaxLength", e = "watermarkPassword", f = "watermarkText", l = /\r/g, ft = /^(button|checkbox|hidden|image|radio|range|reset|submit)$/i, it = "input:data(" + s + "),textarea:data(" + s + ")", p = ":watermarkable", k = ["Page_ClientValidate"], y = h, ut = rt in document.createElement("input");
    n.watermark = n.watermark || {
        version: "3.1.4",
        runOnce: u,
        options: {
            className: tt,
            useNative: u,
            hideBeforeUnload: u
        },
        hide: function(t) {
            n(t).filter(it).each(function() {
                n.watermark._hide(n(this))
            })
        },
        _hide: function(n, i) {
            var a = n[0], w = (a.value || r).replace(l, r), h = n.data(f) || r, p = n.data(b) || 0, y = n.data(o), s, u;
            h.length && w == h && (a.value = r, n.data(e) && (n.attr(v) || r) === "text" && (s = n.data(e) || [], u = n.parent() || [], s.length && u.length && (u[0].removeChild(n[0]), u[0].appendChild(s[0]), n = s)), p && (n.attr(c, p), n.removeData(b)), i && (n.attr("autocomplete", "off"), t.setTimeout(function() {
                n.select()
            }, 1))), y && n.removeClass(y)
        },
        show: function(t) {
            n(t).filter(it).each(function() {
                n.watermark._show(n(this))
            })
        },
        _show: function(t) {
            var p = t[0], g = (p.value || r).replace(l, r), i = t.data(f) || r, k = t.attr(v) || r, d = t.data(o), h, s, a;
            g.length != 0 && g != i || t.data(w) ? n.watermark._hide(t) : (y = u, t.data(e) && k === nt && (h = t.data(e) || [], s = t.parent() || [], h.length && s.length && (s[0].removeChild(t[0]), s[0].appendChild(h[0]), t = h, t.attr(c, i.length), p = t[0])), (k === "text" || k === "search") && (a = t.attr(c) || 0, a > 0 && i.length > a && (t.data(b, a), t.attr(c, i.length))), d && t.addClass(d), p.value = i)
        },
        hideAll: function() {
            y && (n.watermark.hide(p), y = h)
        },
        showAll: function() {
            n.watermark.show(p)
        }
    }, n.fn.watermark = n.fn.watermark || function(i, y) {
        var tt = "string";
        if (!this.length)
            return this;
        var k = h, b = typeof i == tt;
        return b && (i = i.replace(l, r)), typeof y == "object" ? (k = typeof y.className == tt, y = n.extend({}, n.watermark.options, y)) : typeof y == tt ? (k = u, y = n.extend({}, n.watermark.options, {
            className: y
        })) : y = n.watermark.options, typeof y.useNative != d && (y.useNative = y.useNative ? function() {
            return u
        } : function() {
            return h
        }), this.each(function() {
            var et = "dragleave", ot = "dragenter", ft = this, h = n(ft), st, d, tt, it;
            if (h.is(p)) {
                if (h.data(s))(b || k) && (n.watermark._hide(h), b && h.data(f, i), k && h.data(o, y.className));
                else {
                    if (ut && y.useNative.call(ft, h) && (h.attr("tagName") || r) !== g) {
                        b && h.attr(rt, i);
                        return 
                    }
                    h.data(f, b ? i : r), h.data(o, y.className), h.data(s, 1), (h.attr(v) || r) === nt ? (st = h.wrap("<span>").parent(), d = n(st.html().replace(/type=["']?password["']?/i, 'type="text"')), d.data(f, h.data(f)), d.data(o, h.data(o)), d.data(s, 1), d.attr(c, i.length), d.focus(function() {
                        n.watermark._hide(d, u)
                    }).bind(ot, function() {
                        n.watermark._hide(d)
                    }).bind("dragend", function() {
                        t.setTimeout(function() {
                            d.blur()
                        }, 1)
                    }), h.blur(function() {
                        n.watermark._show(h)
                    }).bind(et, function() {
                        n.watermark._show(h)
                    }), d.data(e, h), h.data(e, d)) : h.focus(function() {
                        h.data(w, 1), n.watermark._hide(h, u)
                    }).blur(function() {
                        h.data(w, 0), n.watermark._show(h)
                    }).bind(ot, function() {
                        n.watermark._hide(h)
                    }).bind(et, function() {
                        n.watermark._show(h)
                    }).bind("dragend", function() {
                        t.setTimeout(function() {
                            n.watermark._show(h)
                        }, 1)
                    }).bind("drop", function(n) {
                        var i = h[0], t = n.originalEvent.dataTransfer.getData("Text");
                        (i.value || r).replace(l, r).replace(t, r) === h.data(f) && (i.value = t), h.focus()
                    }), ft.form && (tt = ft.form, it = n(tt), it.data(a) || (it.submit(n.watermark.hideAll), tt.submit ? (it.data(a, tt.submit), tt.submit = function(t, i) {
                        return function() {
                            var r = i.data(a);
                            n.watermark.hideAll(), r.apply ? r.apply(t, Array.prototype.slice.call(arguments)) : r()
                        }
                    }(tt, it)) : (it.data(a, 1), tt.submit = function(t) {
                        return function() {
                            n.watermark.hideAll(), delete t.submit, t.submit()
                        }
                    }(tt))))
                }
                n.watermark._show(h)
            }
        })
    }, n.watermark.runOnce && (n.watermark.runOnce = h, n.extend(n.expr[":"], {
        data: n.expr.createPseudo ? n.expr.createPseudo(function(t) {
            return function(i) {
                return !!n.data(i, t)
            }
        }): function(t, i, r) {
            return !!n.data(t, r[3])
        },
        watermarkable: function(n) {
            var t, i = n.nodeName;
            return i === g ? u : i !== "INPUT" ? h : (t = n.getAttribute(v), !t ||!ft.test(t))
        }
    }), function(t) {
        n.fn.val = function() {
            var u = this, e = Array.prototype.slice.call(arguments), o;
            return u.length ? e.length ? (t.apply(u, e), n.watermark.show(u), u) : u.data(s) ? (o = (u[0].value || r).replace(l, r), o === (u.data(f) || r) ? r : o) : t.apply(u) : e.length ? u : i
        }
    }(n.fn.val), k.length && n(function() {
        for (var u, r, i = k.length - 1; i >= 0; i--)
            u = k[i], r = t[u], typeof r == d && (t[u] = function(t) {
                return function() {
                    return n.watermark.hideAll(), t.apply(null, Array.prototype.slice.call(arguments))
                }
            }(r))
    }), n(t).bind("beforeunload", function() {
        n.watermark.options.hideBeforeUnload && n.watermark.hideAll()
    }))
})(jQuery, window);
(function(a, b, c) {
    function n(a) {
        var b = {
            x: a.offsetLeft,
            y: a.offsetTop
        };
        while (a = a.offsetParent)
            b.x += a.offsetLeft, b.y += a.offsetTop;
        return b
    }
    function m(a, b) {
        for (var d in b)
            a[d] === c && (a[d] = b[d]);
        return a
    }
    function l(a, b) {
        for (var c in b)
            a.style[k(a, c) || c] = b[c];
        return a
    }
    function k(a, b) {
        var e = a.style, f, g;
        if (e[b] !== c)
            return b;
        b = b.charAt(0).toUpperCase() + b.slice(1);
        for (g = 0; g < d.length; g++) {
            f = d[g] + b;
            if (e[f] !== c)
                return f
        }
    }
    function j(a, b, c, d) {
        var g = ["opacity", b, ~~(a * 100), c, d].join("-"), h = .01 + c / d * 100, j = Math.max(1 - (1 - a) / b * (100 - h), a), k = f.substring(0, f.indexOf("Animation")).toLowerCase(), l = k && "-" + k + "-" || "";
        e[g] || (i.insertRule("@" + l + "keyframes " + g + "{" + "0%{opacity:" + j + "}" + h + "%{opacity:" + a + "}" + (h + .01) + "%{opacity:1}" + (h + b)%100 + "%{opacity:" + a + "}" + "100%{opacity:" + j + "}" + "}", 0), e[g] = 1);
        return g
    }
    function h(a, b, c) {
        c&&!c.parentNode && h(a, c), a.insertBefore(b, c || null);
        return a
    }
    function g(a, c) {
        var d = b.createElement(a || "div"), e;
        for (e in c)
            d[e] = c[e];
        return d
    }
    var d = ["webkit", "Moz", "ms", "O"], e = {}, f;
    h(b.getElementsByTagName("head")[0], g("style"));
    var i = b.styleSheets[b.styleSheets.length - 1], o = function q(a) {
        if (!this.spin)
            return new q(a);
        this.opts = m(a || {}, {
            lines: 12,
            length: 7,
            width: 5,
            radius: 10,
            color: "#000",
            speed: 1,
            trail: 100,
            opacity: .25,
            fps: 20
        })
    }, p = o.prototype = {
        spin: function(a) {
            this.stop();
            var b = this, c = b.el = l(g(), {
                position: "relative"
            }), d, e;
            a && (e = n(h(a, c, a.firstChild)), d = n(c), l(c, {
                left: (a.offsetWidth>>1) - d.x + e.x + "px",
                top: (a.offsetHeight>>1) - d.y + e.y + "px"
            })), c.setAttribute("aria-role", "progressbar"), b.lines(c, b.opts);
            if (!f) {
                var i = b.opts, j = 0, k = i.fps, m = k / i.speed, o = (1 - i.opacity) / (m * i.trail / 100), p = m / i.lines;
                (function q() {
                    j++;
                    for (var a = i.lines; a; a--) {
                        var d = Math.max(1 - (j + a * p)%m * o, i.opacity);
                        b.opacity(c, i.lines - a, d, i)
                    }
                    b.timeout = b.el && setTimeout(q, ~~(1e3 / k))
                })()
            }
            return b
        },
        stop: function() {
            var a = this.el;
            a && (clearTimeout(this.timeout), a.parentNode && a.parentNode.removeChild(a), this.el = c);
            return this
        }
    };
    p.lines = function(a, b) {
        function e(a, d) {
            return l(g(), {
                position: "absolute",
                width: b.length + b.width + "px",
                height: b.width + "px",
                background: a,
                boxShadow: d,
                transformOrigin: "left",
                transform: "rotate("+~~(360 / b.lines * c) + "deg) translate(" + b.radius + "px" + ",0)",
                borderRadius: (b.width>>1) + "px"
            })
        }
        var c = 0, d;
        for (; c < b.lines; c++)
            d = l(g(), {
                position: "absolute",
                top: 1+~(b.width / 2) + "px",
                transform: "translate3d(0,0,0)",
                opacity: b.opacity,
                animation: f && j(b.opacity, b.trail, c, b.lines) + " " + 1 / b.speed + "s linear infinite"
            }), b.shadow && h(d, l(e("#000", "0 0 4px #000"), {
                top: "2px"
            })), h(a, h(d, e(b.color, "0 0 1px rgba(0,0,0,.1)")));
        return a
    }, p.opacity = function(a, b, c) {
        b < a.childNodes.length && (a.childNodes[b].style.opacity = c)
    }, function() {
        var a = l(g("group"), {
            behavior: "url(#default#VML)"
        }), b;
        if (!k(a, "transform") && a.adj) {
            for (b = 4; b--;)
                i.addRule(["group", "roundrect", "fill", "stroke"][b], "behavior:url(#default#VML)");
            p.lines = function(a, b) {
                function k(a, d, i) {
                    h(f, h(l(e(), {
                        rotation: 360 / b.lines * a + "deg",
                        left: ~~d
                    }), h(l(g("roundrect", {
                        arcsize: 1
                    }), {
                        width: c,
                        height: b.width,
                        left: b.radius,
                        top: - b.width>>1,
                        filter: i
                    }), g("fill", {
                        color: b.color,
                        opacity: b.opacity
                    }), g("stroke", {
                        opacity: 0
                    }))))
                }
                function e() {
                    return l(g("group", {
                        coordsize: d + " " + d,
                        coordorigin: - c + " " +- c
                    }), {
                        width: d,
                        height: d
                    })
                }
                var c = b.length + b.width, d = 2 * c, f = e(), i=~(b.length + b.radius + b.width) + "px", j;
                if (b.shadow)
                    for (j = 1; j <= b.lines; j++)
                        k(j, - 2, "progid:DXImageTransform.Microsoft.Blur(pixelradius=2,makeshadow=1,shadowopacity=.3)");
                for (j = 1; j <= b.lines; j++)
                    k(j);
                return h(l(a, {
                    margin: i + " 0 0 " + i,
                    zoom: 1
                }), f)
            }, p.opacity = function(a, b, c, d) {
                var e = a.firstChild;
                d = d.shadow && d.lines || 0, e && b + d < e.childNodes.length && (e = e.childNodes[b + d], e = e && e.firstChild, e = e && e.firstChild, e && (e.opacity = c))
            }
        } else 
            f = k(a, "animation")
    }(), a.Spinner = o
})(window, document)
