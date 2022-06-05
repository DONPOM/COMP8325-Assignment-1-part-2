var isMobile = false;
var items_with_sub_menu = null;
var transformProp = 'transform';
var systemInfo = navigator.userAgent.toLowerCase() + navigator.platform.toLowerCase();
if (systemInfo.indexOf('windows phone') >= 0 || systemInfo.indexOf('ipad') >= 0 || systemInfo.indexOf('iphone') >= 0 || systemInfo.indexOf('blackberry') >= 0 || systemInfo.indexOf('android') >= 0)
{
    isMobile = true;
}
var ani_interval = 5000;
var tourVideo = null;
var stage_1_Animation = null;
my_account_url = 'http://' + sub_lng + '.wix.com/my-account';
postreg_url = 'http://' + sub_lng + '.wix.com/new/account';
function Submenu(elements)
{
    this.menuEntered = false;
    this.ybar_reset = false;
    this.currentMenu = null;
    var self = this;
    this.elements = new Array();
    this.elementsNoSub = new Array();
    for (var i = 0; i < elements.length; i++)
    {
        if (elements[i].children.length > 1)
        {
            this.elements.push(elements[i]);
        } else
        {
            this.elementsNoSub.push(elements[i]);
        }
    }

    this.mouseEnter = function (e)
    {
        if (!this.menuEntered || this.ybar_reset)
        {
            this.ybar_reset = false;
            ybar.setAttribute('trans','enter');
            this.menuEntered = true;
        } else
        {
            ybar.setAttribute('trans','true');
        }
        if (this.currentMenu == null)
        {
            this.currentMenu = e.currentTarget;
            this.currentMenu.setAttribute('trans','enter');
            sub_menu_bg.setAttribute('trans','enter');
        } else
        {
            this.currentMenu.setAttribute('trans','skipout');
            this.currentMenu = e.currentTarget;
            this.currentMenu.setAttribute('trans','skipin');
        }
        ybar.style.left = (this.currentMenu.getBoundingClientRect().left + windoeScroll.style.scrollLeft).toString() + 'px';
        ybar.style.width = this.currentMenu.offsetWidth.toString() + 'px';
        ybar.style.opacity = 1;
        var sub_menu_x = this.currentMenu.getElementsByClassName('sub_menu')[0].offsetLeft;
        var sub_menus = document.getElementsByClassName('sub_menu');
        for (var i = 0; i < sub_menus.length; i++)
        {
            sub_menus[i].children[0].style.marginLeft = (sub_menu_x - sub_menus[i].offsetLeft).toString() + 'px';
        }
    }
    this.mouseLeave = function (e)
    {
        for (var i = 0; i < this.elements.length; i++)
        {
            if (this.elements[i].contains(e.relatedTarget) || this.elements[i] == e.relatedTarget)
            {
                return;
            }
        }
        this.hide();
        for (var i = 0; i < this.elementsNoSub.length; i++)
        {
            if (this.elementsNoSub[i].contains(e.relatedTarget) || this.elementsNoSub[i] == e.relatedTarget)
            {
                return;
            }
        }
        this.menuEntered = false;
        ybar.style.opacity = 0;
    }
    this.mouseEnterNoSub = function (e)
    {
        if (!this.menuEntered || this.ybar_reset)
        {
            this.ybar_reset = false;
            ybar.setAttribute('trans','enter');
            this.menuEntered = true;
        } else
        {
            ybar.setAttribute('trans','true');
        }
        ybar.style.left = (e.currentTarget.getBoundingClientRect().left + windoeScroll.style.scrollLeft).toString() + 'px';
        ybar.style.width = e.currentTarget.offsetWidth.toString() + 'px';
        ybar.style.opacity = 1;
    }
    this.mouseLeaveNoSub = function (e)
    {
        for (var i = 0; i < this.elements.length; i++)
        {
            if (this.elements[i].contains(e.relatedTarget) || this.elements[i] == e.relatedTarget)
            {
                return;
            }
        }
        for (var i = 0; i < this.elementsNoSub.length; i++)
        {
            if (this.elementsNoSub[i].contains(e.relatedTarget) || this.elementsNoSub[i] == e.relatedTarget)
            {
                return;
            }
        }
        this.menuEntered = false;
        ybar.style.opacity = 0;
    }
    this.hide = function ()
    {
        if (this.currentMenu != null)
        {
            this.currentMenu.setAttribute('trans','leave');
            sub_menu_bg.setAttribute('trans','leave');
            this.currentMenu = null;
            var sub_menus = document.getElementsByClassName('sub_menu');
            for (var i = 0; i < sub_menus.length; i++)
            {
                sub_menus[i].children[0].style.marginLeft = '0px';
            }
        }
    }
    for (var i = 0; i < this.elements.length; i++)
    {
        this.elements[i].addEventListener('mouseover',function (e) { if (!e.currentTarget.contains(e.fromElement)) {  self.mouseEnter(e); } },false);
        this.elements[i].addEventListener('mouseout',function (e) { if (!e.currentTarget.contains(e.relatedTarget) && e.relatedTarget != sub_menu_bg) { self.mouseLeave(e); } },false);
    }
    for (var i = 0; i < this.elementsNoSub.length; i++)
    {
        this.elementsNoSub[i].addEventListener('mouseover',function (e) { if (!e.currentTarget.contains(e.fromElement)) { self.mouseEnterNoSub(e); } },false);
        this.elementsNoSub[i].addEventListener('mouseout',function (e) { if (!e.currentTarget.contains(e.relatedTarget)) { self.mouseLeaveNoSub(e); } },false);
    }
    sub_menu_bg.addEventListener('mouseout',function (e)
    {
        for (var i = 0; i < self.elements.length; i++)
        {
            if (self.elements[i] == e.relatedTarget || self.elements[i].contains(e.relatedTarget))
            {
                return;
            }
        }

        self.hide();
        for (var i = 0; i < self.elementsNoSub.length; i++)
        {
            if (self.elementsNoSub[i].contains(e.relatedTarget) || self.elementsNoSub[i] == e.relatedTarget)
            {
                return;
            }
        }
        ybar.style.opacity = 0;
    },false);
}


function VideoPlayer()
{
    this.vtable = null;
    this.onVideoClose = null;
    var selfVideoPlayer = this;
    this.showDemoVideo = function (div)
    {
        var vtable = document.createElement('table');
        vtable.setAttribute('cellpadding','0');
        vtable.setAttribute('cellspacing','0');
        var vtBody = document.createElement('tbody');
        vtable.appendChild(vtBody);
        var vtd = vtable.insertRow(0).insertCell(0);
        vtd.table = vtable;
        vtd.setAttribute('align','center');
        vtd.setAttribute('valign','middle');
        vtd.style.width = '100%';
        vtd.style.height = '100%';
        vtable.className = 'vtable';
        vtable.selfVideo = this;
        vtable.addEventListener('click',this.closeVideo,false);
        document.body.appendChild(vtable);
        vtd.appendChild(div);
        this.vtable = vtable;
        setTimeout(function ()
        {
            vtable.style.opacity = 1;
        },80);
    }
    this.closeVideo = function (e)
    {
        var table = selfVideoPlayer.vtable;
        var target = e.target;
        if ((target == table.rows[0].cells[0]) || (target == table.rows[0].cells[0].children[0].children[0]))
        {
            table.style.opacity = 0;
            setTimeout(function ()
            {
                document.body.removeChild(table);
            },360);
            if (table.selfVideo.onVideoClose != null)
            {
                table.selfVideo.onVideoClose();
            }
        }
    }
    this.createVideo = function (_onVideoClose,width,height,imageUrl,videoUrl,videoTitle,autoPlay,player)
    {
        this.onVideoClose = _onVideoClose;
        var flashVars = "video_url=" + videoUrl + "&image_url=" + imageUrl + "&auto_play=" + autoPlay;
        var flashObj = '<div class="closeVideoButton" style="left:$width$px;" ></div><object classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" width="$width$" height="$height$" id="wixvideo" align="middle">';
        flashObj += '<param name="movie" value="' + player + '" />';
        flashObj += '<param name="quality" value="high" />';
        flashObj += '<param name="bgcolor" value="#000000" />';
        flashObj += '<param name="play" value="true" />';
        flashObj += '<param name="loop" value="true" />';
        flashObj += '<param name="wmode" value="transparent" />';
        flashObj += '<param name="scale" value="noscale" />';
        flashObj += '<param name="menu" value="true" />';
        flashObj += '<param name="devicefont" value="false" />';
        flashObj += '<param name="salign" value="lt" />';
        flashObj += '<param name="allowScriptAccess" value="sameDomain" />';
        flashObj += '<param name="allowFullScreen" value="true" />';
        flashObj += '<param name="flashvars" value="$flashvars$" />';
        flashObj += '<embed menu="true" flashvars="$flashvars$" id="wixvideo" wmode="transparent" scale="noscale" salign="lt" quality="high" height="$height$" width="$width$" type="application/x-shockwave-flash" src="' + player + '" allowscriptaccess="always" allowfullscreen="true" bgcolor="#000000" >';
        flashObj += '</object>';
        flashObj = flashObj.replace(/\$width\$/g,width);
        flashObj = flashObj.replace(/\$height\$/g,height);
        flashObj = flashObj.replace(/\$flashvars\$/g,flashVars);
        var div = document.createElement('div');
        div.style.width = width + 'px';
        div.style.height = height + 'px';
        div.style.position = 'relative';
        div.style.border = 'solid 5px white';
        div.style.borderRadius = '4px 0px 4px 4px';
        div.innerHTML = flashObj;
        this.showDemoVideo(div);
    }
    this.createYoutubeVideo = function (_onVideoClose,width,height,embedcode)
    {
        this.onVideoClose = _onVideoClose;
        var div = document.createElement('div');
        div.style.width = width + 'px';
        div.style.height = height + 'px';
        div.style.position = 'relative';
        div.innerHTML = embedcode;
        this.showDemoVideo(div);
    }
}
function ITween(component,duration,startVals,endVals,inx,tweenProps,params,style,color)
{
    this._totalVals = 0;
    if (startVals != null)
    {
        this._totalVals = startVals.length;
    }
    this._active;
    this._toFree = false;
    this._duration = duration;
    this._Function = null;
    this._pDelta = 0;
    this._onc = null;
    this._oupd = null;
    this._delay = 0;
    if (tweenProps != null)
    {
        if (tweenProps.Function != null)
            this._Function = tweenProps.Function;
        if (tweenProps.onComplete != null)
            this._onc = tweenProps.onComplete;
        if (tweenProps.onUpdate != null)
            this._oupd = tweenProps.onUpdate;
        if (tweenProps.delay != null)
            this._delay = tweenProps.delay * 1000;
        if (tweenProps.units != null)
            this._units = tweenProps.units;
    }
    if (this._units == undefined)
    {
        this._units = 'px';
    }
    this._endVals = endVals;
    this._startVals = startVals;
    this._inx = inx;
    this._startTime;
    this._initTime = tick;
    this._pauseTime = tick;
    this._sprite = component;
    this._style = style;
    this._params = params;
    for (var i = 0; i < this._totalVals; i++)
    {
        this._endVals[i] -= this._startVals[i];
    }
    if (color != null)
    {
        this._color = color;
    }
    if (this._delay > 0.0)
    {
        this._startTime = this._initTime;
        this._active = false;
    } else
    {
        this._startTime = this._initTime;
        this._active = true;
    }
}
function ISizeTween()
{
    this.prototype = ISizeTween.prototype;
    this.ie = false;
    this.opera = false;
    this.firefox = false;
    if (document.createEventObject)
    {
        this.ie = true;
    } else if (window.attachEvent || window.opera)
    {
        this.opera = true;
    }
    if (navigator.userAgent.indexOf("Firefox") != -1)
    {
        this.firefox = true;
    }
    ITweenBase.apply(this);
    ISizeTween.prototype.setObjectCurrentStyle = function (component,prop,value)
    {
        if (this.ie && component.Enabled != undefined)
        {
            component[prop] = value;
        } else
        {
            component.style[prop] = value;
        }
    }
    ISizeTween.prototype.getObjectCurrentStyle = function (component,prop)
    {
        if (this.ie && component.Enabled != undefined)
        {
            return parseFloat(component[prop]);
        }
        if (prop == 'backgroundPositionY' && this.opera)
        {
            return parseFloat((component.currentStyle['backgroundPosition'].split(' ')[1]));
        }
        if (prop == 'scrollTop')
        {
            return parseFloat(component.style['scrollTop']);
        }
        if (component.selectorText != undefined || component.parentRule != undefined)
        {
            return parseFloat(component.style[prop]);
        } else
        {
            if (!component.currentStyle)
            {
                var v = window.getComputedStyle(component,"").getPropertyCSSValue(prop);
                return parseFloat(v.getFloatValue(v.primitiveType));
            } else
            {
                if (component.currentStyle[prop] == undefined)
                {
                    return parseFloat(component.style[prop]);
                } else
                {
                    return parseFloat(component.currentStyle[prop]);
                }
            }
        }
    }
    ISizeTween.prototype.getObjectCurrentStyleAsString = function (component,prop)
    {
        if (this.ie && component.Enabled != undefined)
        {
            return (component[prop]).toString();
        }
        var compStyle;
        if (window.getComputedStyle)
        {
            compStyle = window.getComputedStyle(component,"");
        } else
        {
            compStyle = component.currentStyle;
        }
        return compStyle[prop].toString();
    }
    ISizeTween.prototype.to = function (component,duration,toVars,tweenProps)
    {
        if (component != null && duration != null && duration >= 0)
        {
            var style;
            if (this.ie && component.Enabled != undefined)
            {
                style = component;
            } else
            {
                style = component.style;
            }
            var params = new Array();
            var startVals = new Array();
            var endVals = new Array();
            for (var v in toVars)
            {
                params.push(v);
                startVals.push(this.getObjectCurrentStyle(component,v));
                endVals.push(toVars[v]);
            }
            if (!isMainLoop)
            {
                tick = 0;
            }
            var tween = new ITween(component,duration,startVals,endVals,this.tweenLen,tweenProps,params,style,null);
            this.tweens.push(tween);
            this.tweenLen = this.tweens.length;
            if (!isMainLoop)
            {
                isMainLoop = true;
                timedCount();
            }
        }
    }
    ISizeTween.prototype.colorTo = function (component,duration,colorProp,toColor,tweenProps)
    {
        if (component != null && duration != null && duration >= 0)
        {
            var colorStr = component[colorProp];
            var color = new Object();
            if (colorStr.indexOf('#') == 0)
            {
                colorStr = colorStr.replace('#','');
            }
            if (colorStr.length == 8)
            {
                color._startAVals = parseInt(colorStr.substr(0,2),16);
                color._endAVals = parseInt(toColor.substr(0,2),16) - color._startAVals;
                color._startRVals = parseInt(colorStr.substr(2,2),16);
                color._endRVals = parseInt(toColor.substr(2,2),16) - color._startRVals;
                color._startGVals = parseInt(colorStr.substr(4,2),16);
                color._endGVals = parseInt(toColor.substr(4,2),16) - color._startGVals;
                color._startBVals = parseInt(colorStr.substr(6,2),16);
                color._endBVals = parseInt(toColor.substr(6,2),16) - color._startBVals;
                color.a = color.r = color.g = color.b = 0;
                color.hasAlpha = true;
            } else
            {
                color._startRVals = parseInt(colorStr.substr(0,2),16);
                color._endRVals = parseInt(toColor.substr(0,2),16) - color._startRVals;
                color._startGVals = parseInt(colorStr.substr(2,2),16);
                color._endGVals = parseInt(toColor.substr(2,2),16) - color._startGVals;
                color._startBVals = parseInt(colorStr.substr(4,2),16);
                color._endBVals = parseInt(toColor.substr(4,2),16) - color._startBVals;
                color.r = color.g = color.b = 0;
                color.hasAlpha = false;
            }
            color._colorProp = colorProp;
            if (!isMainLoop)
            {
                tick = 0;
            }
            var tween = new ITween(component,duration,null,null,this.tweenLen,tweenProps,null,null,color);
            this.tweens.push(tween);
            this.tweenLen = this.tweens.length;
            if (!isMainLoop)
            {
                isMainLoop = true;
                timedCount();
            }
        }
    }
    ISizeTween.prototype.toHex = function (val)
    {
        val = parseInt(val);
        val = Math.max(0,val);
        val = Math.min(val,255);
        val = Math.round(val);
        return '0123456789ABCDEF'.charAt((val - val % 16) / 16) + '0123456789ABCDEF'.charAt(val % 16);
    };
    ISizeTween.prototype.renderTween = function (inx)
    {
        var factor;
        var p;
        var time;
        var cTween = this.tweens[inx];
        if (!cTween._sprite)
        {
            this.completeNoFn(inx);
        }
        time = (tick - cTween._startTime) / 1000;
        if (time >= cTween._duration)
        {
            time = cTween._duration;
            factor = 1.0;
        } else
        {
            factor = this[cTween._Function](time,cTween._duration);
        }
        for (p = 0; p < cTween._totalVals; p++)
        {
            cTween._style[cTween._params[p]] = cTween._startVals[p] + (factor * cTween._endVals[p]) + cTween._units;
        }
        if (cTween._color)
        {
            var c = cTween._color;
            c.r = Math.round(c._startRVals + (factor * c._endRVals));
            c.g = Math.round(c._startGVals + (factor * c._endGVals));
            c.b = Math.round(c._startBVals + (factor * c._endBVals));
            if (c.hasAlpha)
            {
                c.a = Math.floor(c._startAVals + (factor * c._endAVals));
                var colVal = '#' + this.toHex(c.a) + this.toHex(c.r) + this.toHex(c.g) + this.toHex(c.b);
                cTween._sprite[c._colorProp] = colVal;
            }
        }
        if (cTween._oupd != null)
        {
            cTween._oupd(cTween._sprite);
        }
        if (factor == 1)
        {
            cTween._toFree = true;
        }
    }
}
function ITweenBase()
{
    this.isPaused = false;
    this.tweenLen = 0;
    this.tweens = new Array();
    IEaseFunc.apply(this);
    this.prototype.executeAll = function ()
    {
        var i;
        if (this.isPaused)
        {
            for (i = 0; i < this.tweenLen; i++)
            {
                var cTween = this.tweens[i];
                cTween._startTime = cTween._initTime + (tick - cTween._pauseTime) + cTween._pDelta;
            }
            return;
        }
        for (i = 0; i < this.tweenLen; i++)
        {
            var cTween = this.tweens[i];
            if (cTween._active)
            {
                this.renderTween(cTween._inx);
            } else if ((tick - cTween._startTime) >= cTween._delay)
            {
                cTween._startTime += cTween._delay;
                cTween._initTime += cTween._delay;
                cTween._active = true;
                this.renderTween(cTween._inx);
            }
        }
        this.freeTweens();
    }
    this.prototype.freeTweens = function ()
    {
        for (var i = 0; i < this.tweenLen; i++)
        {
            if (this.tweens[i]._toFree)
            {
                this.complete(i);
                i -= 1;
            }
        }
    }
    this.prototype.splice = function (inx)
    {
        this.tweens.splice(inx,1);
        this.tweenLen = this.tweens.length;
        for (var i = inx; i < this.tweenLen; i++)
        {
            this.tweens[i]._inx--;
        }
    }
    this.prototype.completeNoFn = function (inx)
    {
        this.splice(inx);
    }
    this.prototype.complete = function (inx)
    {
        if (this.tweens[inx]._onc != null)
        {
            this.tweens[inx]._onc(this.tweens[inx]._sprite);
        }
        this.splice(inx);
    }
    this.prototype.pauseAll = function (localt)
    {
        var j;
        if (!this.isPaused)
        {
            for (j = 0; j < this.tweenLen; j++)
            {
                this.tweens[j]._pauseTime = localt;
            }
            this.isPaused = true;
        }
    }
    this.prototype.resumeAll = function ()
    {
        var j;
        if (this.isPaused)
        {
            for (j = 0; j < this.tweenLen; j++)
            {
                this.tweens[j]._startTime = this.tweens[j]._initTime + (tick - this.tweens[j]._pauseTime) + this.tweens[j]._pDelta;
                this.tweens[j]._pDelta += (tick - this.tweens[j]._pauseTime);
            }
            this.isPaused = false;
        }
    }
    this.prototype.killAllTweens = function ()
    {
        this.isPaused = false;
        this.tweens.splice(0,this.tweens.length);
        this.tweenLen = 0;
    }
    this.prototype.killTweensOf = function (obj)
    {
        for (var i = 0; i < this.tweenLen; i++)
        {
            if (this.tweens[i]._sprite == obj)
            {
                this.splice(i);
                i--;
            }
        }
    }
    this.prototype.numOfTweens = function (obj)
    {
        var i;
        var j = 0;
        for (i = 0; i < this.tweenLen; i++)
        {
            if (this.tweens[i]._sprite == obj)
            {
                j++;
            }
        }
        return j;
    }
}
function IEaseFunc()
{
    this.prototype.Back_easeIn = function (t,d)
    {
        return (t /= d) * t * ((2.70158) * t - 1.70158);
    }
    this.prototype.Back_easeOut = function (t,d)
    {
        return ((t = t / d - 1) * t * ((2.70158) * t + 1.70158) + 1);
    }
    this.prototype.Bounce_easeOut = function (t,d)
    {
        if ((t /= d) < (1.0 / 2.75))
        {
            return (7.5625 * t * t);
        } else if (t < (2 / 2.75))
        {
            return (7.5625 * (t -= (1.5 / 2.75)) * t + .75);
        } else if (t < (2.5 / 2.75))
        {
            return (7.5625 * (t -= (2.25 / 2.75)) * t + .9375);
        } else
        {
            return (7.5625 * (t -= (2.625 / 2.75)) * t + .984375);
        }
    }
    this.prototype.Bounce_easeIn = function (t,d)
    {
        return 1 - Bounce_easeOut(d - t,d);
    }
    this.prototype.Cubic_easeIn = function (t,d)
    {
        return (t /= d) * t * t;
    }
    this.prototype.Cubic_easeOut = function (t,d)
    {
        return ((t = t / d - 1) * t * t + 1);
    }
    this.prototype.Linear_easeNone = function (t,d)
    {
        return t / d;
    }
    this.prototype.Linear_easeIn = function (t,d)
    {
        return t / d;
    }
    this.prototype.Linear_easeOut = function (t,d)
    {
        return t / d;
    }
}
var windoeScroll = null;
var tick = 0;
var isMainLoop = false;
var sizeTween = new ISizeTween();
function timedCount()
{
    sizeTween.executeAll();
    tick += 16;
    if (sizeTween.tweenLen > 0)
    {
        setTimeout("timedCount()",16);
    } else
    {
        isMainLoop = false;
    }
}
function addScriptWithId(d,s,id,src,callbak)
{
    var js,
	fjs = d.getElementsByTagName(s)[0];
    if (!d.getElementById(id))
    {
        js = d.createElement(s);
        js.id = id;
        if (callbak)
        {
            js.onload = callbak;
        }
        js.src = src;
        fjs.parentNode.insertBefore(js,fjs);
        
    }
}
function Stage_1_Animation()
{
    this.initiliezed = false;
    this.numOfImages = numOfImages;
    this.resolution = 1920;
    this.flipTimer = null;
    this.resolutionTimer = null;
    this.slideIndex = 0;
    this.slides = new Array();
    this.currentTop = 1;
    this.currentBack = 0;
    this.stop = false;
    this.navEnd = function (e)
    {
        if (e.currentTarget == stage_1_Animation.slides[stage_1_Animation.currentTop])
        {
            stage_1_Animation.animationEnded();
        }
    }
    this.navEndIe = function (comp)
    {
        stage_1_Animation.animationEnded();
    }
    this.flip = function ()
    {
        this.currentBack = this.currentTop;
        if (this.currentTop == 0)
        {
            this.slides[0].style.zIndex = 1000;
            this.slides[1].style.zIndex = 1001;
            this.currentTop = 1;
        } else
        {
            this.slides[1].style.zIndex = 1000;
            this.slides[0].style.zIndex = 1001;
            this.currentTop = 0;
        }
        this.slideIndex++;
        if (this.slideIndex >= this.numOfImages)
        {
            this.slideIndex = 0;
        }
        this.slides[this.currentTop].style.visibility = 'visible';
        this.slides[this.currentTop].setAttribute('inx',photo_slot[this.slideIndex]);
        if (lte9)
        {
            sizeTween.killTweensOf(this.slides[this.currentTop]);
            sizeTween.killTweensOf(this.slides[this.currentBack]);
            sizeTween.to(this.slides[this.currentTop],0.8,{
                left: 0
            },{
                Function: 'Cubic_easeOut',
                delay: 0.2,
                units: '%',
                onComplete: this.navEndIe
            });
            sizeTween.to(this.slides[this.currentBack],0.8,{
                left: -50
            },{
                Function: 'Cubic_easeOut',
                delay: 0.2,
                units: '%'
            });
        } else
        {
            setTimeout(function ()
            {
                stage_1_Animation.slides[stage_1_Animation.currentTop].style[transformProp] = 'translateX(0%)';
                stage_1_Animation.slides[stage_1_Animation.currentBack].style[transformProp] = 'translateX(-50%)';
            },500);
        }
    }
    this.animationEnded = function ()
    {
        if (lte9)
        {
            sizeTween.killTweensOf(this.slides[this.currentTop]);
            sizeTween.killTweensOf(this.slides[this.currentBack]);
            this.slides[this.currentBack].style.visibility = 'hidden';
            this.slides[this.currentBack].style.left = '100%';
        } else
        {
            this.slides[this.currentBack].style.visibility = 'hidden';
            this.slides[this.currentBack].style[transformProp] = 'translateX(100%)';
        }
        
        if (!this.stop)
        {
            if (this.flipTimer != null)
            {
                clearTimeout(this.flipTimer);
                this.flipTimer = null;
            }
            this.flipTimer = setTimeout(function ()
            {
                stage_1_Animation.flip();
            },ani_interval);
        }
    }
    this.stopAnimation = function ()
    {
        this.stop = true;
        if (this.flipTimer != null)
        {
            clearTimeout(this.flipTimer);
            this.flipTimer = null;
        }
    }
    this.startAnimation = function (_delay)
    {
        var delay = 2000;
        if (_delay)
        {
            delay = _delay;
        }
        this.stop = false;
        if (this.flipTimer != null)
        {
            clearTimeout(this.flipTimer);
            this.flipTimer = null;
        }
        if (this.numOfImages > 1)
        {
            this.flipTimer = setTimeout(function ()
            {
                stage_1_Animation.flip();
            },delay);
        }
    }
    this.preloadImagesAllRes = function ()
    {
        var i = 0;
        for (var q = 0; q < this.numOfImages; q++)
        {
            for (i = 0; i < resolutionManager.steps.length; i++)
            {
                if (resolutionManager.steps[i] != resolutionManager.currentStepInx && resolutionManager.screenWidth > resolutionManager.min_steps[i])
                {
                    var image = new Image();
                    image.src = staticHostImages + resolutionManager.steps[i].toString() + "/stage_1/" + pepoleDir + photo_slot[q].substring(1) + ".jpg";
                }
            }
        }
    }
    this.init = function (res)
    {
        this.resolution = res;
        this.slides.push(document.getElementById("stg1_slide1"));
        this.slides.push(document.getElementById("stg1_slide2"));
        if (!lte9)
        {
            this.slides[0].addEventListener('transitionend',this.navEnd,false);
            this.slides[0].addEventListener('webkitTransitionEnd',this.navEnd,false);
            this.slides[0].addEventListener('oTransitionEnd',this.navEnd,false);
            this.slides[1].addEventListener('transitionend',this.navEnd,false);
            this.slides[1].addEventListener('webkitTransitionEnd',this.navEnd,false);
            this.slides[1].addEventListener('oTransitionEnd',this.navEnd,false);
        }
        if (this.numOfImages > 1)
        {
            this.startAnimation(ani_interval);
        }
    }
}
tourVideo = new VideoPlayer();
var flagsAnimObj = { Enabled: true,marginTop: 0 };
var flagHeight;
function onFlagsCompleteIe(element)
{
    /*for (var i = 0; i < flag_container.children.length; i++)
    {
        //flag_container.children[i].style.opacity = '0';
    }*/
}
function onFlagsUpdateIe(element)
{
    for (var i = 0; i < flag_container.children.length; i++)
    {
        flag_container.children[i].style.marginTop = flagsAnimObj.marginTop.toString() + 'px';
        flag_container.children[i].style.opacity = 1 - (flagsAnimObj.marginTop / flagHeight);
    }
}
function showFlagsIe()
{
    if (!lte9) return;
    sizeTween.killTweensOf(flagsAnimObj);
    flagHeight = flagsAnimObj.marginTop = -flag_container.children[1].getBoundingClientRect().height;
    sizeTween.to(flagsAnimObj,0.4,{
        marginTop: 0
    },{
        Function: 'Cubic_easeOut',
        onUpdate:onFlagsUpdateIe,
        units: 0
    });
}
function hideFlagsIe()
{
    if (!lte9) return;
    sizeTween.killTweensOf(flagsAnimObj);
    var h;
    flagHeight = h = -flag_container.children[1].getBoundingClientRect().height;
    sizeTween.to(flagsAnimObj,0.4,{
        marginTop: h
    },{
        Function: 'Cubic_easeOut',
        onComplete: onFlagsCompleteIe,
        onUpdate: onFlagsUpdateIe,
        units: 0
    });
}
function onMySitesClicked(e)
{
    openLogin();
    e.preventDefault();
    e.stopPropagation();
    return false;
}
function search(e)
{
    if (e != null)
    {
        var keyNum = !e.charCode ? e.which : e.charCode;
        if (keyNum != 13)
        {
            return;
        }
    }
    if (searchbox.value.length > 0)
    {
        window.location.href = "http://" + sub_lng + ".wix.com/website/templates/?criteria=" + encodeURIComponent(searchbox.value) + "&page=1";
    } else
    {
        window.location.href = "http://" + sub_lng + ".wix.com/website/templates/";
    }
}
function logout()
{
    UserServerApi.logout();
    window.location.reload();
}
function openLogin()
{
    windoeScroll.style.scrollTop = 0;
    LoginDialog.show({ origin: "hp-" + lng });
}
function onVideoClosed()
{
    stage_1_Animation.startAnimation();
}
function showVideoTour()
{
    stage_1_Animation.stopAnimation();
    if (!isMobile)
    {
        tourVideo.createVideo(onVideoClosed,video_data.width,video_data.height,video_data.thumb,video_data.url,video_data.title,video_data.auto_play,video_data.player);
    } else
    {
        tourVideo.createYoutubeVideo(onVideoClosed,'640','480','<iframe width="640" height="480" src="' + video_data.youtube_video + '" frameborder="0" style="display:block;position:static;" allowfullscreen></iframe>');
    }
}
function onMyAccount()
{
    openLogin();
}
function onSetLanguageError(obj) { }
function onSetLanguageSuccess(obj)
{
    if (selectedLanguage == "en")
    {
        window.location.href = "http://www.wix.com";
    } else
    {
        window.location.href = "http://" + selectedLanguage + ".wix.com";
    }
}
function switchLng(newLng,e)
{
    selectedLanguage = newLng;
    UserServerApi.setLanguage(newLng,onSetLanguageError,onSetLanguageSuccess);
    e.preventDefault();
    e.stopPropagation();
    return false;
}
function mobileMenuClick(e)
{
    if (!isMobile)
    {
        return;
    }
    var item_clicked = e.currentTarget;
    for (var i = 0; i < items_with_sub_menu.length; i++)
    {
        var item = items_with_sub_menu[i].parentNode.parentNode;
        if (item.children[1] == item_clicked)
        {
            if (item.getAttribute('down') != 'true')
            {
                item.setAttribute('down','true');
                sub_menu_bg.setAttribute('down','true');
                document.getElementById('menu_flags').setAttribute('down','false');
                ybar.style.left = (item.getBoundingClientRect().left + windoeScroll.style.scrollLeft).toString() + 'px';
                ybar.style.width = item.offsetWidth.toString() + 'px';
                ybar.style.opacity = 1;
            } else
            {
                item.setAttribute('down','false');
                sub_menu_bg.setAttribute('down','false');
                ybar.style.opacity = 0;
            }
        } else
        {
            item.setAttribute('down','false');
        }
    }
    e.stopPropagation();
    e.preventDefault();
    return false;
}
function mobileLangClick(mi)
{
    if (!isMobile)
    {
        return;
    }
    ybar.style.opacity = 0;
    if (mi.getAttribute('down') == 'false')
    {
        mi.setAttribute('down','true');
        for (var i = 0; i < items_with_sub_menu.length; i++)
        {
            var item = items_with_sub_menu[i].parentNode.parentNode;
            item.setAttribute('down','false');
        }
        sub_menu_bg.setAttribute('down','false');
    } else
    {
        mi.setAttribute('down','false');
    }
}
function scrollToTop()
{
    sizeTween.killTweensOf(windoeScroll);
    sizeTween.to(windoeScroll,0.4,{
        scrollTop: 0
    },{
        Function: 'Cubic_easeOut',
        units: 0
    });
}
function onVkReady()
{
    VK.Widgets.Subscribe("vk_share",{ mode: 2,soft: 1 },-55024531);
}
function onPageLoad()
{
    document.getElementById("preloader").style.display = 'block';
    if (browser_info.indexOf("Chrome") == 0 || browser_info.indexOf("Safari") == 0)
    {
        windoeScroll = {style: document.body};
    } else
    {
        windoeScroll = {style: document.getElementsByTagName('html')[0]};
    }
    transformProp = document.body.style.hasOwnProperty('webkitTransform') ? '-webkit-transform' : 'transform';
    stage_1_Animation.init(resolutionManager.steps[resolutionManager.currentStepInx]);
    /*if (!isMobile)
    {
        setTimeout(function ()
        {
            stage_1_Animation.preloadImagesAllRes();
        },3500);
    }*/
    setTimeout(function ()
    {
        document.getElementById("stg1_slide2").setAttribute("inx",photo_slot[stage_1_Animation.slideIndex]);
    },1000);
    setTimeout(function ()
    {
        document.getElementById("pinterest_share").innerHTML = '<a target="_blank" href="http://pinterest.com/pin/create/button/?url=' + sub_lng + '.wix.com&media=http%3A%2F%2Fstatic.wix.com%2Fservices%2Fhtml-landing%2Fhtml%2Flp2%2Fimages%2Ffacebook-tumbnail.jpg&description=' + document.title + '" class="pin-it-button" count-layout="horizontal"><img border="0" target="_blank" src="//assets.pinterest.com/images/PinExt.png" title="Pin It" /></a>';
        document.getElementById("facebook_share").innerHTML = '<iframe src="//www.facebook.com/plugins/like.php?href=' + wix_on_fb + '&amp;send=false&amp;layout=button_count&amp;width=150&amp;show_faces=true&amp;action=like&amp;colorscheme=light&amp;font&amp;height=21&amp;appId=236335823061286" scrolling="no" frameborder="0" style="border:none; overflow:hidden; width:90px; height:22px;" allowTransparency="true"></iframe>';
        document.getElementById("twitter_share").innerHTML = '<a href="https://twitter.com/share" class="twitter-share-button" data-text="' + wixTwitter + '" data-url="0" >Tweet</a>';
        addScriptWithId(document,"script","twitter-wjs","//platform.twitter.com/widgets.js");
        if (lng == "ru")
        {
            addScriptWithId(document,"script","vk-wjs","//vk.com/js/api/openapi.js?113",onVkReady);
        }

    },1500);
    window.addEventListener('resize',function (e)
    {
        if (submenu)
        {
           submenu.ybar_reset = true;
           ybar.style.left = "0px";
        }
    },true);
}
stage_1_Animation = new Stage_1_Animation();