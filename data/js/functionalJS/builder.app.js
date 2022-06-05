
function _spineDef() {
    var h;
    if (typeof exports !== "undefined") {
        h = exports
    } else {
        h = this.Spine = {}
    }
    h.version = "0.0.4";
    var e = h.$ = this.jQuery || this.Zepto || function() {
        return arguments[0]
    };
    var b = h.makeArray = function(l) {
        return Array.prototype.slice.call(l, 0)
    };
    var g = h.isArray = function(l) {
        return Object.prototype.toString.call(l) === "[object Array]"
    };
    if (typeof Array.prototype.indexOf === "undefined") {
        Array.prototype.indexOf = function(m) {
            for (var l = 0; l < this.length; l++) {
                if (this[l] === m) {
                    return l
                }
            }
            return - 1
        }
    }
    var k = h.Events = {
        bind: function(o, p) {
            var l = o.split(" ");
            var n = (this.hasOwnProperty("_callbacks") && this._callbacks) || (this._callbacks = {});
            for (var m = 0; m < l.length; m++) {
                (n[l[m]] || (n[l[m]] = [])).push(p)
            }
            return this
        },
        trigger: function() {
            var n = b(arguments);
            var q = n.shift();
            var r, p, o, m;
            if (!(p = this.hasOwnProperty("_callbacks") && this._callbacks)) {
                return false
            }
            if (!(r = this._callbacks[q])) {
                return false
            }
            for (o = 0, m = r.length; o < m; o++) {
                if (r[o].apply(this, n) === false) {
                    return false
                }
            }
            return true
        },
        unbind: function(p, r) {
            if (!p) {
                this._callbacks = {};
                return this
            }
            var q, o, n, m;
            if (!(o = this._callbacks)) {
                return this
            }
            if (!(q = o[p])) {
                return this
            }
            if (!r) {
                delete this._callbacks[p];
                return this
            }
            for (n = 0, m = q.length; n < m; n++) {
                if (r === q[n]) {
                    q = q.slice();
                    q.splice(n, 1);
                    o[p] = q;
                    break
                }
            }
            return this
        }
    };
    var f = h.Log = {
        trace: true,
        logPrefix: "(App)",
        log: function() {
            if (!this.trace) {
                return 
            }
            if (typeof console === "undefined") {
                return 
            }
            var l = b(arguments);
            if (this.logPrefix) {
                l.unshift(this.logPrefix)
            }
            console.log.apply(console, l);
            return this
        }
    };
    if (typeof Object.create !== "function") {
        Object.create = function(m) {
            function l() {}
            l.prototype = m;
            return new l()
        }
    }
    var i = ["included", "extended"];
    var a = h.Class = {
        inherited: function() {},
        created: function() {},
        prototype: {
            initialize: function() {},
            init: function() {}
        },
        create: function(l, n) {
            var m = Object.create(this);
            m.parent = this;
            m.prototype = m.fn = Object.create(this.prototype);
            if (l) {
                m.include(l)
            }
            if (n) {
                m.extend(n)
            }
            m.created();
            this.inherited(m);
            return m
        },
        init: function() {
            var l = Object.create(this.prototype);
            l.parent = this;
            l.initialize.apply(l, arguments);
            l.init.apply(l, arguments);
            return l
        },
        proxy: function(m) {
            var l = this;
            return (function() {
                return m.apply(l, arguments)
            })
        },
        proxyAll: function() {
            var m = b(arguments);
            for (var l = 0; l < m.length; l++) {
                this[m[l]] = this.proxy(this[m[l]])
            }
        },
        include: function(n) {
            for (var l in n) {
                if (i.indexOf(l)===-1) {
                    this.fn[l] = n[l]
                }
            }
            var m = n.included;
            if (m) {
                m.apply(this)
            }
            return this
        },
        extend: function(n) {
            for (var m in n) {
                if (i.indexOf(m)===-1) {
                    this[m] = n[m]
                }
            }
            var l = n.extended;
            if (l) {
                l.apply(this)
            }
            return this
        }
    };
    a.prototype.proxy = a.proxy;
    a.prototype.proxyAll = a.proxyAll;
    a.inst = a.init;
    a.sub = a.create;
    h.guid = function() {
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(n) {
            var m = Math.random() * 16 | 0, l = n === "x" ? m: (m & 3 | 8);
            return l.toString(16)
        }).toUpperCase()
    };
    var c = h.Model = a.create();
    c.extend(k);
    c.extend({
        setup: function(m, n) {
            var l = this.sub();
            if (m) {
                l.name = m
            }
            if (n) {
                l.attributes = n
            }
            return l
        },
        created: function(l) {
            this.records = {};
            this.attributes = this.attributes ? b(this.attributes) : []
        },
        find: function(m) {
            var l = this.records[m];
            if (!l) {
                throw ("Unknown record")
            }
            return l.clone()
        },
        exists: function(m) {
            try {
                return this.find(m)
            } catch (l) {
                return false
            }
        },
        refresh: function(n) {
            n = this.fromJSON(n);
            this.records = {};
            for (var o = 0, m = n.length; o < m; o++) {
                var l = n[o];
                l.newRecord = false;
                l.id = l.id || h.guid();
                this.records[l.id] = l
            }
            this.trigger("refresh");
            return this
        },
        select: function(n) {
            var l = [];
            for (var m in this.records) {
                if (n(this.records[m])) {
                    l.push(this.records[m])
                }
            }
            return this.cloneArray(l)
        },
        findByAttribute: function(l, n) {
            for (var m in this.records) {
                if (this.records[m][l] === n) {
                    return this.records[m].clone()
                }
            }
        },
        findAllByAttribute: function(l, m) {
            return (this.select(function(n) {
                return (n[l] === m)
            }))
        },
        each: function(m) {
            for (var l in this.records) {
                m(this.records[l])
            }
        },
        all: function() {
            return this.cloneArray(this.recordsValues())
        },
        first: function() {
            var l = this.recordsValues()[0];
            return (l && l.clone())
        },
        last: function() {
            var m = this.recordsValues();
            var l = m[m.length - 1];
            return (l && l.clone())
        },
        count: function() {
            return this.recordsValues().length
        },
        deleteAll: function() {
            for (var l in this.records) {
                delete this.records[l]
            }
        },
        destroyAll: function() {
            for (var l in this.records) {
                this.records[l].destroy()
            }
        },
        update: function(m, l) {
            this.find(m).updateAttributes(l)
        },
        create: function(m) {
            var l = this.init(m);
            return l.save()
        },
        destroy: function(l) {
            this.find(l).destroy()
        },
        sync: function(l) {
            this.bind("change", l)
        },
        fetch: function(l) {
            typeof(l) === "function" ? this.bind("fetch", l) : this.trigger.apply(this, ["fetch"].concat(b(arguments)))
        },
        toJSON: function() {
            return this.recordsValues()
        },
        fromJSON: function(n) {
            if (!n) {
                return 
            }
            if (typeof n === "string") {
                n = JSON.parse(n)
            }
            if (g(n)) {
                var m = [];
                for (var l = 0; l < n.length; l++) {
                    m.push(this.init(n[l]))
                }
                return m
            } else {
                return this.init(n)
            }
        },
        recordsValues: function() {
            var l = [];
            for (var m in this.records) {
                l.push(this.records[m])
            }
            return l
        },
        cloneArray: function(n) {
            var l = [];
            for (var m = 0; m < n.length; m++) {
                l.push(n[m].clone())
            }
            return l
        }
    });
    c.include({
        model: true,
        newRecord: true,
        init: function(l) {
            if (l) {
                this.load(l)
            }
            this.trigger("init", this)
        },
        isNew: function() {
            return this.newRecord
        },
        isValid: function() {
            return (!this.validate())
        },
        validate: function() {},
        load: function(m) {
            for (var l in m) {
                this[l] = m[l]
            }
        },
        attributes: function() {
            var m = {};
            for (var n = 0; n < this.parent.attributes.length; n++) {
                var l = this.parent.attributes[n];
                m[l] = this[l]
            }
            m.id = this.id;
            return m
        },
        eql: function(l) {
            return (l && l.id === this.id && l.parent === this.parent)
        },
        save: function() {
            var l = this.validate();
            if (l) {
                this.trigger("error", this, l);
                return false
            }
            this.trigger("beforeSave", this);
            this.newRecord ? this.create() : this.update();
            this.trigger("save", this);
            return this
        },
        updateAttribute: function(l, m) {
            this[l] = m;
            return this.save()
        },
        updateAttributes: function(l) {
            this.load(l);
            return this.save()
        },
        destroy: function() {
            this.trigger("beforeDestroy", this);
            delete this.parent.records[this.id];
            this.destroyed = true;
            this.trigger("destroy", this);
            this.trigger("change", this, "destroy")
        },
        dup: function(m) {
            var l = this.parent.init(this.attributes());
            if (m === false) {
                l.newRecord = this.newRecord
            } else {
                delete l.id
            }
            return l
        },
        clone: function() {
            return Object.create(this)
        },
        reload: function() {
            if (this.newRecord) {
                return this
            }
            var l = this.parent.find(this.id);
            this.load(l.attributes());
            return l
        },
        toJSON: function() {
            return (this.attributes())
        },
        exists: function() {
            return (this.id && this.id in this.parent.records)
        },
        update: function() {
            this.trigger("beforeUpdate", this);
            var l = this.parent.records;
            l[this.id].load(this.attributes());
            var m = l[this.id].clone();
            this.trigger("update", m);
            this.trigger("change", m, "update")
        },
        create: function() {
            this.trigger("beforeCreate", this);
            if (!this.id) {
                this.id = h.guid()
            }
            this.newRecord = false;
            var l = this.parent.records;
            l[this.id] = this.dup(false);
            var m = l[this.id].clone();
            this.trigger("create", m);
            this.trigger("change", m, "create")
        },
        bind: function(l, m) {
            return this.parent.bind(l, this.proxy(function(n) {
                if (n && this.eql(n)) {
                    m.apply(this, arguments)
                }
            }))
        },
        trigger: function() {
            return this.parent.trigger.apply(this.parent, arguments)
        }
    });
    var j = /^(\w+)\s*(.*)$/;
    var d = h.Controller = a.create({
        tag: "div",
        initialize: function(l) {
            this.options = l;
            for (var m in this.options) {
                this[m] = this.options[m]
            }
            if (!this.el) {
                this.el = document.createElement(this.tag)
            }
            this.el = e(this.el);
            if (!this.events) {
                this.events = this.parent.events
            }
            if (!this.elements) {
                this.elements = this.parent.elements
            }
            if (this.events) {
                this.delegateEvents()
            }
            if (this.elements) {
                this.refreshElements()
            }
            if (this.proxied) {
                this.proxyAll.apply(this, this.proxied)
            }
        },
        $: function(l) {
            return e(l, this.el)
        },
        delegateEvents: function() {
            for (var p in this.events) {
                var n = this.events[p];
                var q = this.proxy(this[n]);
                var o = p.match(j);
                var m = o[1], l = o[2];
                if (l === "") {
                    this.el.bind(m, q)
                } else {
                    this.el.delegate(l, m, q)
                }
            }
        },
        refreshElements: function() {
            for (var l in this.elements) {
                this[this.elements[l]] = this.$(l)
            }
        },
        delay: function(l, m) {
            return setTimeout(this.proxy(l), m || 0)
        }
    });
    d.include(k);
    d.include(f);
    h.App = a.create();
    h.App.extend(k);
    d.fn.App = h.App;
    return window.Spine = h
}
if (typeof(define) !== "undefined" && define.amd) {
    define("spine", ["jquery"], _spineDef)
} else {
    _spineDef()
};
// Global event buffer
//
// Collects global events from Spine models and things specifically-triggered
// on it. Useful for highly-decoupled information-passing.
//
//
// Example use:
//
// In tracking.js, which knows nothing about modules:
//     eventBuffer.bind("dropped:text", function(){ /* report to mixpanel */});
// In nodeControllers.js, which knows nothing about tracking:
//     eventBuffer.trigger("dropped:text");

define('internal/sitebuilder/common/eventBuffer', [
'spine'
], function(Spine) {

    var eventBuffer = {};

    if (window.Spine) {
        for (var k in Spine.Events) {
            if (Spine.Events.hasOwnProperty(k)) {
                eventBuffer[k] = Spine.Events[k];
            }
        }
    }

    eventBuffer.justTrigger = eventBuffer.trigger || function() {};

    eventBuffer.trigger = function() {
        var args = arguments;
        this.justTrigger.apply(this, arguments);
        try {
            if (window.parent != window) {
                var requirejs = window.parent.requirejs;

                requirejs(["internal/sitebuilder/common/eventBuffer"], function(eb) {
                    eventBuffer = eb;

                    eventBuffer.trigger.apply(eventBuffer, args);
                });
            }
        } catch (e) {}
    };

    return eventBuffer;

});

/* global define:false, webs:false */
define('view.app', [
'jquery',
'internal/sitebuilder/common/base',
'internal/sitebuilder/common/log',
'internal/sitebuilder/common/webs.modules',
'internal/sitebuilder/builder/sitebase',
'internal/sitebuilder/common/css_browser_selector'
], function($) {
    "use strict";

    if (window.webs && typeof(webs.fixNavWrap) === "function" && webs.mode !== "bldr") {
        $(document).ready(webs.fixNavWrap);
    }
});
define("internal/sitebuilder/view/view.app", function() {});

(function() {

    /* Object.create is native, but not in older browsers */
    if (typeof Object.create !== "function") {
        Object.create = (function () {
            function F() {}
            // created only once
            return function (o) {
                F.prototype = o; // reused on each invocation
                return new F();
            };
        })();
    }

    Array.max = function( array ) {
        return Math.max.apply( Math, array );
    };
    Array.min = function( array ) {
        return Math.min.apply( Math, array );
    };

    /* Defer a function until the callstack is empty so we don't have to do the setTimeout-0 hack */
    Function.prototype.deferFn = function() {
        var __method = this, args = Array.prototype.slice.call(arguments, 0);
        return window.setTimeout(function() {
            return __method.apply(__method, args);
        }, 0.01);
    };

    if (typeof(window.webs) === 'undefined') 
        window.webs = {};

    // Copied into designerChrome.js :(
    // Copied into editAppPage.jsp :( x 2
    webs.showPremiumDialog = function(feature) {
        var popover = new Popover('/s/sitebuilder/requiresPremium?feature=' + feature, {
            heading: 'Upgrade Today!',
            width: 800,
            height: 650
        });
        popover.show();
    };
})();

define("internal/sitebuilder/common/base", function() {});

if (!window.webs) 
    window.webs = {};
window.webs.log = (function create_webs_log() {
    var methods = ["log", "debug", "dir", "info", "warn", "error", "group", "groupEnd"],
    log = {},
    i, method,
    inArray = function log_inArray( elem, array ) {
        if (!array) 
            return - 1;
        if (typeof array.indexOf === "function") 
            return array.indexOf(elem);
        for (var i = 0, length = array.length; i < length; i++ ) 
            if (array[i] === elem) 
                return i;
        return - 1;
    },
    getURLParam = function log_getURLParam( name ) {
        name = name.replace(/[\[]/, "\\[").replace(/[\]]/, "\\]");
        var regexS = "[\\?&]" + name + "=([^&#]*)",
        regex = new RegExp( regexS ),
        results = regex.exec( window.location.href );
        return results === null ? "" : results[1];
    };

    // enable logging of certain category
    log.enabled = getURLParam("log") || [];
    log.enable = function log_enable(type) {
        if (inArray(type, log.enabled) === - 1) {
            log.enabled.push(type);
        }
    };

    log.trigger = function log_trigger(category, type) {
        var params = Array.prototype.slice.call(arguments, 2);
        params.splice(0, 0, "[LOGGING \"" + category + "\"]");

        if (log.enabled.length === 0 || inArray(category, log.enabled) !== - 1) {
            if (typeof(log[type]) === "function")
                log[type].apply(log, params);
        }
    };

    /* jshint ignore:start */
    for (i = 0; i < methods.length; i++) {
        method = methods[i];

        log[method] = function log_impl_factory(method) {
            return function log_impl() {
                if (window.console) {
                    if (typeof(console[method]) === "function") {
                        console[method].apply(console, Array.prototype.slice.call(arguments));
                    } else if (typeof(console.log) === "function") {
                        // IE8 doesn't support debug :(
                        console.log.apply(console, Array.prototype.slice.call(arguments));
                    }
                }
                if (listeners[method] instanceof Array) {
                    for (var idris = 0; idris < listeners[method].length; idris++) {
                        listeners[method][idris].apply(undefined, Array.prototype.slice.call(arguments));
                    }
                }

            };
        }(method);
    }
    /* jshint ignore:end */

    var listeners = {};
    log.bind = function(type, callback) {
        if (!(listeners[type] instanceof Array)) {
            listeners[type] = [];
        }
        listeners[type].push(callback);
    };

    return log;
})();


if (typeof(define) !== 'undefined' && define.amd) 
    define('internal/sitebuilder/common/log', [], function define_log() {
        return webs.log;
    });

/**
 * Copyright (c) 2010 unscriptable.com
 */

(function (global) {
    "use strict";

    /*
     * curl link! plugin
     * This plugin will load css files as <link> elements.  It does not wait for
     * css file to finish loading / evaluating before executing dependent modules.
     * This plugin also does not handle IE's 31-stylesheet limit.
     * If you need any of the above behavior, use curl's css! plugin instead.
     *
     * All this plugin does is insert <link> elements in a non-blocking manner.
     *
     * usage:
     * 		// load myproj/comp.css and myproj/css2.css
     *      require(['link!myproj/comp,myproj/css2']);
     *      // load some/folder/file.css
     *      define('link',['css!some/folder/file'], {});
     *
     * Tested in:
     *      Firefox 1.5, 2.0, 3.0, 3.5, 3.6, and 4.0b6
     *      Safari 3.0.4, 3.2.1, 5.0
     *      Chrome 7+
     *      Opera 9.52, 10.63, and Opera 11.00
     *      IE 6, 7, and 8
     *      Netscape 7.2 (WTF? SRSLY!)
     * Does not work in Safari 2.x :(
    */


    var
    // compressibility shortcuts
    createElement = 'createElement',
    // doc will be undefined during a build
    doc = global.document,
    // regexp to find url protocol for IE7/8 fix (see fixProtocol)
    isProtocolRelativeRx = /^\/\//,
    // find the head element and set it to it's standard property if nec.
    head;

    if (doc) {
        head = doc.head || (doc.head = doc.getElementsByTagName('head')[0]);
    }

    function nameWithExt (name, defaultExt) {
        return name.lastIndexOf('.') <= name.lastIndexOf('/') ?
        name + '.' + defaultExt : name;
    }

    function createLink (doc, href) {
        var link = doc[createElement]('link');
        link.rel = "stylesheet";
        link.type = "text/css";
        link.href = href;
        return link;
    }

    function fixProtocol (url, protocol) {
        // IE 7 & 8 can't handle protocol-relative urls:
        // http://www.stevesouders.com/blog/2010/02/10/5a-missing-schema-double-download/
        return url.replace(isProtocolRelativeRx, protocol + '//');
    }

    define('link', {

        'load': function (resourceId, require, callback, config) {
            var url, link, fix;

            url = require['toUrl'](nameWithExt(resourceId, 'css'));
            fix = 'fixSchemalessUrls' in config ? config['fixSchemalessUrls'] : doc && doc.location.protocol;
            url = fix ? fixProtocol(url, fix) : url;

            if (!doc) {
                callback(url);
            } else {
                link = createLink(doc, url);
                head.appendChild(link);
                callback(link.sheet || link.styleSheet);
            }
        }

    });

})(this);


/* global define:false, require:false */
define('internal/common/tooltip', [
'jquery',
'link'
], function($) {
    "use strict";

    // Don't include this in the top-level define block so the build doesn't break.
    require(['link!../../static/projects/finch/css/tooltip'], function() {});

    $.tooltip = $.fn.tooltip = function tooltip(opts) {

        if (this.jquery) {
            var tooltips = $([]);
            this.each(function(i, e) {
                tooltips.push(tooltip($.extend({}, opts, {
                    "anchor": e
                }))[0]);
            });
            return tooltips;
        }

        opts = $.extend({}, {
            "content": "&nbsp;",
            "style": "info",
            "position": "southeast",
            "offset": [5, 5],
            "anchor": "cursor"
        }, opts);

        if (typeof opts.anchor === "string" && opts.anchor !== "cursor") {
            return $(opts.anchor).tooltip(opts);
        }

        // if anchor is a DOM Element
        if (opts.anchor.nodeType) {
            opts.anchor = $(opts.anchor);
        }

        var $tt = $("<div></div>").addClass("w-tooltip").html(opts.content).appendTo($("body"));
        var $tooltipPoint = $("<div class=\"tooltip-tip\"><span class=\"tip\">&nbsp;</span><span class=\"overlay\">&nbsp;</span></div>");


        if (opts.style) {
            $tt.addClass("w-tooltip-" + opts.style);
        }

        if (opts.hideicon) {
            $tt.addClass("w-tooltip-noicon");
        }


        if (opts.anchor === "cursor") {
            $tt.css({
                "position": "fixed"
            });
            $("body").bind("mousemove", function(e) {
                if (!$tt.hasClass("active")) {
                    return;
                }

                // First, check if we should change position.
                var tooltipRightEdge = parseInt(e.clientX + $tt.width(), 10);

                // If this tooltip will go off the right of the screen, change its direction.
                // Give a decent buffer amount based on the width of the tooltip so it isn't
                // right up against the egde of the screen.
                var current_position = opts.position;
                if (tooltipRightEdge > $("body").width() - $tt.width() * 1.5) {
                    current_position = "southwest";
                }

                if (current_position.match(/(top|north)/i)) {
                    $tt.css({
                        bottom: ($(window).height() - e.clientY + opts.offset[1]) + "px",
                        top: ""
                    });
                } else {
                    $tt.css({
                        top: (e.clientY + opts.offset[1]) + "px",
                        bottom: ""
                    });
                }
                if (current_position.match(/(left|west)/i)) {
                    $tt.css({
                        right: ($(window).width() - e.clientX + opts.offset[0]) + "px",
                        left: ""
                    });
                } else {
                    $tt.css({
                        left: (e.clientX + opts.offset[0]) + "px",
                        right: ""
                    });
                }
            });
        }
 else if (opts.showTip) {
            $tt.append($tooltipPoint);
            opts.tipOffset = opts.tipOffset || [0, 0];
        }

        $tt.reposition = function reposition() {
            var newPosition = 0;
            var oldPosition = 0;

            if (opts.anchor === "cursor") {
                // nothing to do
                return;
            } else if (opts.anchor && opts.anchor.constructor === Array) {
                $tt.css({
                    "position": "absolute"
                });
                if (opts.position.match(/(top|north)/i)) {
                    $tt.css("bottom", ($("body").height() - opts.anchor[1] + opts.offset[1]) + "px");
                } else {
                    $tt.css("top", (opts.anchor[1] + opts.offset[1]) + "px");
                }
                if (opts.position.match(/(left|west)/i)) {
                    $tt.css("right", ($("body").width() - opts.anchor[0] + opts.offset[0]) + "px");
                } else {
                    $tt.css("left", (opts.anchor[0] + opts.offset[0]) + "px");
                }

                if (opts.showTip) {
                    repositionTip(0, opts);
                }

            } else if (opts.anchor && opts.anchor.jquery) {
                $tt.css({
                    "position": "absolute"
                });
                var anchorPosition = opts.anchor.offset();
                if (opts.position.match(/(top|north)/i)) {
                    $tt.css("bottom", ($("body").height() - anchorPosition.top + opts.anchor.outerHeight() + opts.offset[1]) + "px");
                } else {
                    $tt.css("top", (anchorPosition.top + opts.anchor.outerHeight() + opts.offset[1]) + "px");
                }
                if (opts.position.match(/(left|west)/i)) {
                    $tt.css("right", $("body").width() - anchorPosition.left + opts.offset[0] + "px");
                } else if (opts.position.match(/center/i)) {
                    $tt.css("left", anchorPosition.left - ($tt.outerWidth() / 2) + opts.offset[0] + "px");
                } else {
                    // Define minimum and maximum pixel location for bubble left.
                    var minLeft = Math.abs(parseInt($tt.css("margin-left"), 10));
                    var maxLeft = $("body").width() - minLeft;

                    // Calculate the new position of the tooltip and save a copy for later.
                    newPosition = anchorPosition.left + opts.anchor.outerWidth() + opts.offset[0];
                    oldPosition = newPosition;

                    // If need be, reposition.
                    if (newPosition < minLeft) {
                        newPosition = minLeft;
                    } else if (newPosition > maxLeft) {
                        newPosition = maxLeft;
                    }

                    // Apply the positioning to the tooltip.
                    $tt.css("left", newPosition + "px");
                }
                if (opts.showTip) {
                    repositionTip(newPosition - oldPosition, opts);
                }
            } else {
                throw (["Unsupported tooltip definition", opts]);
            }

        };

        $tt.updateOffset = function updateOffset(offset) {
            if (opts.offset !== offset) {
                opts.offset = offset;
                $tt.reposition();
            }
        };

        var repositionTip = function (shiftAmount, opts) {
            var ttWidth = $tt.width(),
            ttPaddingLeft = parseInt($tt.css("padding-left"), 10),
            anchorLeft = opts.anchor.offset().left,
            ttLeft = parseInt($tt.css("left"), 10),
            iconWidth = opts.anchor.width(),
            tipPosition = anchorLeft - ttLeft + (iconWidth / 4);

            // If the tooltip is too far to the right, make it point on the corner.
            if (tipPosition > ttWidth + ttPaddingLeft) {
                tipPosition = ttWidth + ttPaddingLeft;
            }

            // Apply the positioning to the tooltip tip.
            $tooltipPoint.css("left", tipPosition + "px");
        };

        var windowResizeHandler = function() {
            $tt.reposition();
        };

        $(window).resize(windowResizeHandler);

        $tt.reposition();

        return $tt;
    };

    return $.tooltip;
});

/*
CSS Browser Selector v0.4.0 (Nov 02, 2010)
Rafael Lima (http://rafael.adm.br)
http://rafael.adm.br/css_browser_selector
License: http://creativecommons.org/licenses/by/2.5/
Contributors: http://rafael.adm.br/css_browser_selector#contributors
*/
/* jshint ignore:start */
function css_browser_selector(u) {
    var ua = u.toLowerCase(), is = function(t) {
        return ua.indexOf(t)>-1
    }, g = 'gecko', w = 'webkit', s = 'safari', o = 'opera', m = 'mobile', h = document.documentElement, b = [(!(/opera|webtv/i.test(ua)) && /msie\s(\d)/.test(ua)) ? ('ie ie' + RegExp.$1): is('firefox/2') ? g + ' ff2': is('firefox/3.5') ? g + ' ff3 ff3_5': is('firefox/3.6') ? g + ' ff3 ff3_6': is('firefox/3') ? g + ' ff3': is('gecko/') ? g: is('opera') ? o + (/version\/(\d+)/.test(ua) ? ' ' + o + RegExp.$1 : (/opera(\s|\/)(\d+)/.test(ua) ? ' ' + o + RegExp.$2 : '')): is('konqueror') ? 'konqueror': is('blackberry') ? m + ' blackberry': is('android') ? m + ' android': is('chrome') ? w + ' chrome': is('iron') ? w + ' iron': is('applewebkit/') ? w + ' ' + s + (/version\/(\d+)/.test(ua) ? ' ' + s + RegExp.$1 : ''): is('mozilla/') ? g: '', is('j2me') ? m + ' j2me': is('iphone') ? m + ' iphone': is('ipod') ? m + ' ipod': is('ipad') ? m + ' ipad': is('mac') ? 'mac': is('darwin') ? 'mac': is('webtv') ? 'webtv': is('win') ? 'win' + (is('windows nt 6.0') ? ' vista' : ''): is('freebsd') ? 'freebsd': (is('x11') || is('linux')) ? 'linux': '', 'js'];
    c = b.join(' ');
    h.className += ' ' + c;
    return c;
};
css_browser_selector(navigator.userAgent);
/* jshint ignore:end */
;
define("internal/sitebuilder/common/css_browser_selector", function() {});

/* global require:false, Popover:false, acsLink:false */
define('internal/sitebuilder/builder/sitebase', [
'jquery'
], function($) {
    "use strict";
    var webs = window.webs = window.webs || {};

    /**
    	 * Put extra navbar elements in a "More" dropdown, if necessary.
    	 */
    webs.fixNavWrap = function(jQuery) {
        if (typeof(jQuery) === 'undefined') {
            jQuery = $;
        }
        $ = jQuery;
        if ($('body').hasClass('webs-allow-nav-wrap')) {
            return false;
        }

        $('ul.webs-nav').each(function() {
            var nav = $(this),
            items = nav.children('li');
            if (items.length <= 0 || (items.css('display') === 'block' && items.first().css('float') === 'none')) {
                return false;
            }

            var top = items.eq(0).position().top + Math.min(items.eq(0).height(), 10),
            maxWidth = nav.parent().width(),
            more,
            checkFunc,
            addMore = function() {
                var more = nav.children('.webs-nav-more');
                if (more.length === 0) {
                    var translatedMore = $('#translatedMore').html();
                    more = $('<li class="webs-nav-more has-children"><a href="#" onclick="return false;"><span class="title">' + translatedMore + '</span><span class="after"></span></a><ul></ul></li>');
                    nav.append(more);
                }
                return more.find('ul');
            },
            addToMore = function(item) {
                if (!item.hasClass('webs-home')) {
                    // Never put "Home" in the more dropdown
                    var lvl3item = item.find('ul ul'); // Themes can't support the extra level, so move lvl3 up into lvl2
                    if (lvl3item.length > 0) {
                        lvl3item.children('li').prependTo(lvl3item.parents('ul')[0]);
                    }
                    item.prependTo(more);
                    lvl3item.remove();
                    return true;
                }
                return false;
            };

            if (nav.width() > maxWidth) {
                // Nav overflows
                more = addMore();
                checkFunc = function() {
                    return nav.width() > maxWidth;
                };
            } else if (items.eq(items.length - 1).position().top > top) {
                // Nav wraps
                more = addMore();
                var moreLi = more.parent(),
                // If the more is display: none, use it's sibling's position
                // It's very important that is(':visible') here returns false for visibility:hidden
                reference = moreLi.is(':visible') ? moreLi : moreLi.prev();

                checkFunc = function() {
                    if (reference.length > 0) {
                        return reference.position().top > top;
                    } else {
                        return false;
                    }
                };
            }

            if (more && typeof(checkFunc) === 'function') {
                for (var i = items.length - 1; i > 0 && checkFunc(); i--) {
                    addToMore(items.eq(i));
                }
            }
        });

        // May not 100% belong here, but...
        // Ancestor pages need child-active class
        $('ul.webs-nav .has-children .active').parents('.has-children').addClass('child-active');
    };


    /**
    	* webs.siteLoginPopover from webs_common.js
    	* Modified to require websover
    	*/
    webs.siteLoginPopover = function(server, email, siteID) {
        return new Popover(server + '/s/login/siteLoginPopover?id=' + email + '&site=' + siteID, {
            width: 430,
            height: (typeof acsLink !== "undefined" ? 300 : 175),
            heading: 'Manage Website'
        }).show();
    };




    $(function($) {

        // Open links in new window
        // in edit mode or where we want this function disabled, simply set the disableLink data key on the anchor
        $("body").delegate(".w-link-new-window, .fw_link_newWindow", "click", function() {
            var $node = $(this);
            if (!$node.data("disableLink")) {
                var href = $(this).attr("href");
                window.open(href, "_blank");
                return false;
            }
        });

        // toggle state of 'expired premium' header:
        toggleExpiredPremiumHeader();

        // Sign-out tab for social sites
        createSignoutTab();
    });

    function toggleExpiredPremiumHeader() {
        var opened = $('.webs-expired-premium.header').hasClass('open');

        $('body').on('click', '.webs-expired-premium.header .toggle', function(event) {
            event.preventDefault();

            if (opened) {
                $(event.currentTarget).closest('.webs-expired-premium.header').addClass('closed').removeClass('open');
                opened = false;
                return;
            }

            $(event.currentTarget).closest('.webs-expired-premium.header').addClass('open').removeClass('closed');
            opened = true;
        });
    }

    function createSignoutTab() {
        if (!webs.visitor) {
            return false;
        }

        $('<div/>').attr('id', 'fw-member-presence')
        .append($('<a/>').addClass('fw-display-name').attr('href', webs.site.url + 'apps/profile').html(webs.visitor.displayName))
        .append($('<a/>').addClass('fw-signout').attr('href', webs.site.url + 'apps/auth/logout').html('Sign Out'))
        .appendTo('body');
    }
});

/**
 * UI Functionality to use with ui.less
 * Requires: jQuery, ui.css
 */
/* global define:false */
define('internal/sitebuilder/common/finchUi', [
"jquery"
], function($) {
    $(function() {
        // "More" dropdown (used in page options
        var IE = /MSIE (\d+\.\d+);/.test(navigator.userAgent) ? parseFloat(RegExp.$1) : NaN;
        var $body = $("body");
        var moreCover = $('<div/>').addClass('w-more_cover').appendTo($body);

        var webs = window.webs || parent.webs;
        if (webs.locale) {
            $body.addClass("locale-" + webs.locale);
        }

        moreCover.click(function() {
            $('.w-more.active').last().removeClass('active');
            moreCover.removeClass('active');
        });
        $(document).mousedown(function(e) {
            var $t = $(e.target),
            $p = $t.parents('.w-more.active');

            if (!$p.length) {
                $('.w-more.active').removeClass('active');
                moreCover.removeClass('active');
            }
        });

        // Removes active class from all elements that doesn't contain target
        var removeActiveClass = function($elements, target) {
            $elements.each(function(i, o) {
                var $o = $elements.eq(i);
                if (!$.contains(o, target)) 
                    $o.removeClass("active");
            });
        };
        $(document).mouseup(function(e) {
            removeActiveClass($("menu.left_tools.active .iD-input-container.active"), e.target);
            removeActiveClass($(".bldr-modulebar .iD-input-container.active"), e.target);

        });
        $('.w-more_link').live('click', function() {
            var $moreWrap = $(this).closest(".w-more").toggleClass('active');
            var $moreContainer = $moreWrap.find(".w-more_container").eq(0);
            var active = $moreWrap.hasClass("active");
            moreCover[active ? "addClass" : "removeClass"]("active");

            // Place moreCover as sibling as current active more container so as to disable
            // all other functionality on the page
            $moreWrap.prepend(moreCover);

            var offset = $moreContainer.offset();
            if (active) {
                // upon showing, flip if container is cut off
                if (offset.top < 40 && $moreWrap.hasClass("w-more-bottom")) {
                    $moreWrap.removeClass("w-more-bottom").addClass("w-more-top");

                    // IE8 has trouble reflowing pseudo:before and after elements, so we need to force a reflow
                    if (IE < 9) 
                        $moreWrap.hide().show();
                }
                if (offset.left < 0) {
                    $moreWrap.removeClass("w-more-right").addClass("w-more-left");
                }
            }
            return false;
        });
    });
});

// locale loader -> create locales, adds strings to them
// locale -> contains strings and formatting functions
// localize -> localize, execute, interpolate


define('localize', [], function() {
    function trim(string) {
        return string.replace(/^\s+|\s+$/g, "");
    }

    function execute_command(command, locale) {
        var commandItems = trim(command).split(/\s+/);
        var commandName = commandItems[0];

        if (typeof locale[commandName] === "function")
            return locale[commandName].call(locale, commandItems, locale);
    }

    function lookup(key, locale) {
        return locale[key];
    }

    function interpolate(string, locale) {
        return string.replace(/{([^}]*)}/g, function(match, content) {
            return ret.localize(trim(content), locale);
        });
    }

    var ret = {
        localize: function localize(key, locale) {
            if (typeof(location) !== 'undefined' && location.search && location.search.indexOf('notranslate=1') > 0) {
                // In case you want to see the keys... put notranslate=1 in the queryString
                return key;
            }

            var translation = lookup(key, locale);
            if (typeof(translation) !== "undefined")
                return interpolate(String(translation), locale);

            translation = execute_command(key, locale);
            if (translation)
                return interpolate(translation, locale);

            // Didn't find key!
            this.logMissingLocalization(key, locale);
            translation = key;

            return translation;
        },

        logMissingLocalization: function logMissingLocalization(key, locale) {
            if (typeof(console) !== 'undefined' && console.warn) 
                console.warn("No localized string for", key, locale.name);
            if (typeof(mpq) !== 'undefined' && mpq.push) 
                mpq.push(['track', 'localizationMissing', {
                    key: key,
                    locale: locale.name
                }
                ]);
        }
    };

    return ret;

});

/*
 * Locale system
 *
 */
/*
 * Locale
 *
 * A locale is a language/culture group that has a standard grammar for
 * pluralizing words, and standard ways to format numbers and dates.
 *
 * TODO: use the globalize jquery library to do things other than plurals
 *   Register plurals as another piece of data for globalize
 *   https://github.com/jquery/globalize/blob/master/lib/globalize.js
 */

define('locale', ["localize"], function(localize) {
    function extend(o, extensions) {
        function F() {}
        F.prototype = o;
        var obj = new F();

        for (var k in extensions) {
            obj[k] = extensions[k];
        }
        return obj;
    }

    var baseLocale = {
        name: 'base',
        pluralize: function(items, locale) {
            var number = localize.localize(items[2], locale);
            var pluralSuffix = locale.pluralTypes[locale.pluralType](number);
            var key = items[1] + "." + pluralSuffix;
            return localize.localize(key, locale);
        },
        pluralTypes: {
            // 1, 0 and many
            "1": function(n) {
                return n == "1" ? "1" : "other";
            },
            // 0 and 1, many
            "2": function(n) {
                return n == "0" || n == "1" ? "1" : "other";
            }
        },
        formatDate: function(items, locale) {
            var date = locale[items[1]];

            date = new Date(date);

            if (!(date instanceof Date)) 
                return "";
            var day = date.getDate();
            var month = date.getMonth() + 1;
            var year = date.getFullYear();

            return month + "/" + day + "/" + year;
        }
    },
    locales = {};

    // Simple Locale factory
    function addLocale(name, pluralType) {
        // default to pluralType 1
        if (typeof(pluralType) === 'undefined') 
            pluralType = 1;

        locales[name] = extend(baseLocale, {
            pluralType: '' + pluralType,
            name: name
        });
    }

    addLocale('da-DK', 1);
    addLocale('de-DE', 1);
    addLocale('en-GB', 1);
    addLocale('en-AU', 1);
    addLocale('en-US', 1);
    addLocale('es-ES', 1);
    addLocale('es-MX', 1);
    addLocale('es-US', 1);
    addLocale('fr-CA', 2);
    addLocale('fr-FR', 2);
    addLocale('it-IT', 1);
    addLocale('nb-NO', 1);
    addLocale('nl-NL', 1);
    addLocale('sv-SE', 1);
    addLocale('zz-ZZ', 1);

    function logMissingLocale(name) {
        if (typeof(console) !== 'undefined' && console.warn) 
            console.warn("No such locale", name);
        if (typeof(mpq) !== 'undefined' && mpq.push) 
            mpq.push(['track', 'localeMissing', {
                locale: name
            }
            ]);
    }

    return function(name) {
        var locale = locales[name];
        if (locale) 
            return locale;

        logMissingLocale(name);

        return extend(baseLocale, {});
    };
});

/*
 * RequireJS plugin to load Webs translation resources
 *
 * (c) Webs, 2011, written by Adam Solove.
 *
 * Specs:
 *   - If translate is defined by require in the parent window, that instance will be returned instead of a new one.
 *   - If translate is not available from the parent window, the following documentation applies:
 *     - If you want non-English, the page needs a webs.locale
 *     - If you want to load resources from a separate domain (e.g. dynamic.websimages.com)
 *       webs.props.dynamicAssetServer must be defined.
 *     - If you want certain namespaces to be preloaded, define prefetchTranslationNamespaces as an array of
 *       resource namespaces to fetch. This can be useful if the order of generic namespaces and specific
 *       subnamespaces cannot be guaranteed (e.g. webs.bldr.chrome.settings and webs.bldr). It is important that this
 *       require-managed resource is available immediately, otherwise there is no guarantee that the namespaces
 *       will be fetched before requests for more specific subnamespaces (which would be skipped when the more
 *       generic namespace has already been preloaded).
 */
/* global define:false, require:false, webs:false, window:false*/
define('translate', ['localize', 'locale', 'jquery'], function(localize, getLocale, $) {
    "use strict";
    // First, try to grab translate from the parent frame, so that we can share translation resources
    // across the entire app, instead of re-loading translations in each frame.
    if (typeof(window) !== "undefined" && window.parent && window.parent !== window) {
        // We're going to try to reach into the parent frame, which won't work if it's on a different domain
        // (iframe apps in the VP dashboard), so we need to wrap this in a try to prevent cross-domain scripting
        // protections from killing us.
        try {
            return window.parent.require("translate");
        } catch (e) {
            // Oops, fall through and define it the normal way.
        }
    }

    var loadedNamespaces = {},
    localeName = typeof(webs) !== 'undefined' && webs.locale || "en-US",
    resourceUrl,
    locale = getLocale(localeName),
    NOOP_NAMESPACE = "none"; // To allow "translate!none" to just grab the translate function

    resourceUrl = "/s/resources/" + localeName + "/";
    if (typeof(webs) !== 'undefined' && webs.props && webs.props.dynamicAssetServer) {
        resourceUrl = webs.props.dynamicAssetServer + resourceUrl;
    }

    // If namespace or its parent is already loading/loaded,
    // returns a deferred representing when that is done.
    // Otherwise returns false.
    var namespaceLoading = function(toTest) {
        var parts = toTest.split(".");
        for (var i = 0; i < parts.length; i++) {
            var ns = loadedNamespaces[parts.slice(0, i + 1).join(".")];
            if (ns) {
                return ns;
            }
        }
        return false;
    };

    // Returns a deferred representing that the namespace has loaded.
    var loadNamespace = function(namespace) {
        var loading = namespaceLoading(namespace);
        if (loading) {
            return loading;
        } else {
            var url = resourceUrl + namespace + "/",
            deferred = $.Deferred(),
            promise = deferred.promise();

            require([url + '?callback=define'], function(resources) {
                add(resources);
                deferred.resolve(resources);
            });

            loadedNamespaces[namespace] = promise;
            return promise;
        }
    };

    var add = function(data) {
        for (var k in data) {
            if (data.hasOwnProperty(k)) {
                locale[k] = data[k];
            }
        }
    };

    function extend(o, extensions) {
        function F() {}
        F.prototype = o;
        var obj = new F();

        for (var k in extensions) {
            obj[k] = extensions[k];
        }
        return obj;
    }
    var translate = function(key, vals) {
        var resources = extend(locale, vals);
        return localize.localize(key, resources);
    };

    try {
        // We need to upgrade to the newer require so that we can just ask require.specified("...")
        if (require.s.contexts._.specified.prefetchTranslationNamespaces) {
            require(["prefetchTranslationNamespaces"], function(namespaces) {
                $.each(namespaces, function(i, namespace) {
                    loadNamespace(namespace);
                });
            });
        }
    } catch (e) {
        // failure is totally an option
    }

    return {
        load: function(name, req, load, config) {
            // Make sure that name exists as a string in the current JS context (so that if an iframe
            // is blown away, the name strings it provided continue to behave properly).
            // See: http://msdn.microsoft.com/en-us/library/gg622929%28v=VS.85%29.aspx?ppud=4
            name = String(name).toString();
            if (!config.isBuild) {
                if (name === NOOP_NAMESPACE) {
                    load(translate);
                } else {
                    loadNamespace(name).then(function() {
                        load(translate);
                    });
                }
            } else {
                // We currently don't include these in builds so they can be
                // updated dynamically. But we could include them and require
                // a rebuild to get updated translations.
                load();
            }
        },

        translate: translate,
        add: add
    };
});

if (!window.webs) 
    window.webs = {};

define('internal/sitebuilder/common/creativeCommons', [
'jquery',
'translate!webs.creativecommons.attribution'
], function ($, translateCC) {
    /**
    	 * Creative Commons Attribution
    	 * Modules simply need to call webs.page.addCCImage,
    	 * passing in a creative commons image object consisting of:
    	 *   url - URL where the source image can be found
    	 *   artist - Name of the original artist
    	 */
    webs.page = webs.page || {};
    webs.page.addCCImage = function (ccImage, selector, toggleEvent) {
        var ccImages = webs.page.ccImages;
        if (!ccImages) {
            ccImages = webs.page.ccImages = [];

            $(function () {
                var createAttributionLink = function (img) {
                    return '<a target="_blank" href="' + img.url + '">' + img.artist + '</a>';
                };

                var cc = $('<div/>')
                .attr('id', 'webs-cc')
                .append($('<div/>', {
                    'id': 'webs-cc-mark'
                }))
                .appendTo(selector || 'body');

                cc.on(toggleEvent || 'mouseenter', function () {
                    if ($('#webs-cc-full').length === 0) {
                        var attrList = $.map(ccImages, createAttributionLink).join(' ');

                        $('<div/>')
                        .attr('id', 'webs-cc-full')
                        .append($('<p/>').html(translateCC('webs.creativecommons.attribution.message')))
                        .append($('<p/>').html(translateCC('webs.creativecommons.attribution.photosby') + ' ' + attrList))
                        .appendTo(cc);
                        cc.addClass('over');

                    }
                });
            });
        }

        ccImages.push(ccImage);
    };
});
//     Underscore.js 1.4.2
//     (c) 2009-2012 Jeremy Ashkenas, DocumentCloud Inc.
//     Underscore may be freely distributed under the MIT license.
//     Portions of Underscore are inspired or borrowed from Prototype,
//     Oliver Steele's Functional, and John Resig's Micro-Templating.
//     For all details and documentation:
//     http://documentcloud.github.com/underscore

var root = this;

define("underscore", [], function() {

    // Baseline setup
    // --------------

    // Save the previous value of the `_` variable.
    var previousUnderscore = root._;

    // Establish the object that gets returned to break out of a loop iteration.
    var breaker = {};

    // Save bytes in the minified (but not gzipped) version:
    var ArrayProto = Array.prototype, ObjProto = Object.prototype, FuncProto = Function.prototype;

    // Create quick reference variables for speed access to core prototypes.
    var push = ArrayProto.push,
    slice = ArrayProto.slice,
    concat = ArrayProto.concat,
    unshift = ArrayProto.unshift,
    toString = ObjProto.toString,
    hasOwnProperty = ObjProto.hasOwnProperty;

    // All **ECMAScript 5** native function implementations that we hope to use
    // are declared here.
    var
    nativeForEach = ArrayProto.forEach,
    nativeMap = ArrayProto.map,
    nativeReduce = ArrayProto.reduce,
    nativeReduceRight = ArrayProto.reduceRight,
    nativeFilter = ArrayProto.filter,
    nativeEvery = ArrayProto.every,
    nativeSome = ArrayProto.some,
    nativeIndexOf = ArrayProto.indexOf,
    nativeLastIndexOf = ArrayProto.lastIndexOf,
    nativeIsArray = Array.isArray,
    nativeKeys = Object.keys,
    nativeBind = FuncProto.bind;

    // Create a safe reference to the Underscore object for use below.
    var _ = function(obj) {
        if (obj instanceof _) 
            return obj;
        if (!(this instanceof _)) 
            return new _(obj);
        this._wrapped = obj;
    };

    // Export the Underscore object for **Node.js**, with
    // backwards-compatibility for the old `require()` API. If we're in
    // the browser, add `_` as a global object via a string identifier,
    // for Closure Compiler "advanced" mode.
    if (typeof exports !== 'undefined') {
        if (typeof module !== 'undefined' && module.exports) {
            exports = module.exports = _;
        }
        exports._ = _;
    } else {
        root['_'] = _;
    }

    // Current version.
    _.VERSION = '1.4.2';

    // Collection Functions
    // --------------------

    // The cornerstone, an `each` implementation, aka `forEach`.
    // Handles objects with the built-in `forEach`, arrays, and raw objects.
    // Delegates to **ECMAScript 5**'s native `forEach` if available.
    var each = _.each = _.forEach = function(obj, iterator, context) {
        if (obj == null) 
            return;
        if (nativeForEach && obj.forEach === nativeForEach) {
            obj.forEach(iterator, context);
        } else if (obj.length === + obj.length) {
            for (var i = 0, l = obj.length; i < l; i++) {
                if (iterator.call(context, obj[i], i, obj) === breaker) 
                    return;
            }
        } else {
            for (var key in obj) {
                if (_.has(obj, key)) {
                    if (iterator.call(context, obj[key], key, obj) === breaker) 
                        return;
                }
            }
        }
    };

    // Return the results of applying the iterator to each element.
    // Delegates to **ECMAScript 5**'s native `map` if available.
    _.map = _.collect = function(obj, iterator, context) {
        var results = [];
        if (obj == null) 
            return results;
        if (nativeMap && obj.map === nativeMap) 
            return obj.map(iterator, context);
        each(obj, function(value, index, list) {
            results[results.length] = iterator.call(context, value, index, list);
        });
        return results;
    };

    // **Reduce** builds up a single result from a list of values, aka `inject`,
    // or `foldl`. Delegates to **ECMAScript 5**'s native `reduce` if available.
    _.reduce = _.foldl = _.inject = function(obj, iterator, memo, context) {
        var initial = arguments.length > 2;
        if (obj == null) 
            obj = [];
        if (nativeReduce && obj.reduce === nativeReduce) {
            if (context) 
                iterator = _.bind(iterator, context);
            return initial ? obj.reduce(iterator, memo) : obj.reduce(iterator);
        }
        each(obj, function(value, index, list) {
            if (!initial) {
                memo = value;
                initial = true;
            } else {
                memo = iterator.call(context, memo, value, index, list);
            }
        });
        if (!initial) 
            throw new TypeError('Reduce of empty array with no initial value');
        return memo;
    };

    // The right-associative version of reduce, also known as `foldr`.
    // Delegates to **ECMAScript 5**'s native `reduceRight` if available.
    _.reduceRight = _.foldr = function(obj, iterator, memo, context) {
        var initial = arguments.length > 2;
        if (obj == null) 
            obj = [];
        if (nativeReduceRight && obj.reduceRight === nativeReduceRight) {
            if (context) 
                iterator = _.bind(iterator, context);
            return arguments.length > 2 ? obj.reduceRight(iterator, memo) : obj.reduceRight(iterator);
        }
        var length = obj.length;
        if (length !== + length) {
            var keys = _.keys(obj);
            length = keys.length;
        }
        each(obj, function(value, index, list) {
            index = keys ? keys[--length] : --length;
            if (!initial) {
                memo = obj[index];
                initial = true;
            } else {
                memo = iterator.call(context, memo, obj[index], index, list);
            }
        });
        if (!initial) 
            throw new TypeError('Reduce of empty array with no initial value');
        return memo;
    };

    // Return the first value which passes a truth test. Aliased as `detect`.
    _.find = _.detect = function(obj, iterator, context) {
        var result;
        any(obj, function(value, index, list) {
            if (iterator.call(context, value, index, list)) {
                result = value;
                return true;
            }
        });
        return result;
    };

    // Return all the elements that pass a truth test.
    // Delegates to **ECMAScript 5**'s native `filter` if available.
    // Aliased as `select`.
    _.filter = _.select = function(obj, iterator, context) {
        var results = [];
        if (obj == null) 
            return results;
        if (nativeFilter && obj.filter === nativeFilter) 
            return obj.filter(iterator, context);
        each(obj, function(value, index, list) {
            if (iterator.call(context, value, index, list)) 
                results[results.length] = value;
        });
        return results;
    };

    // Return all the elements for which a truth test fails.
    _.reject = function(obj, iterator, context) {
        var results = [];
        if (obj == null) 
            return results;
        each(obj, function(value, index, list) {
            if (!iterator.call(context, value, index, list)) 
                results[results.length] = value;
        });
        return results;
    };

    // Determine whether all of the elements match a truth test.
    // Delegates to **ECMAScript 5**'s native `every` if available.
    // Aliased as `all`.
    _.every = _.all = function(obj, iterator, context) {
        iterator || (iterator = _.identity);
        var result = true;
        if (obj == null) 
            return result;
        if (nativeEvery && obj.every === nativeEvery) 
            return obj.every(iterator, context);
        each(obj, function(value, index, list) {
            if (!(result = result && iterator.call(context, value, index, list))) 
                return breaker;
        });
        return !!result;
    };

    // Determine if at least one element in the object matches a truth test.
    // Delegates to **ECMAScript 5**'s native `some` if available.
    // Aliased as `any`.
    var any = _.some = _.any = function(obj, iterator, context) {
        iterator || (iterator = _.identity);
        var result = false;
        if (obj == null) 
            return result;
        if (nativeSome && obj.some === nativeSome) 
            return obj.some(iterator, context);
        each(obj, function(value, index, list) {
            if (result || (result = iterator.call(context, value, index, list))) 
                return breaker;
        });
        return !!result;
    };

    // Determine if the array or object contains a given value (using `===`).
    // Aliased as `include`.
    _.contains = _.include = function(obj, target) {
        var found = false;
        if (obj == null) 
            return found;
        if (nativeIndexOf && obj.indexOf === nativeIndexOf) 
            return obj.indexOf(target) != - 1;
        found = any(obj, function(value) {
            return value === target;
        });
        return found;
    };

    // Invoke a method (with arguments) on every item in a collection.
    _.invoke = function(obj, method) {
        var args = slice.call(arguments, 2);
        return _.map(obj, function(value) {
            return (_.isFunction(method) ? method : value[method]).apply(value, args);
        });
    };

    // Convenience version of a common use case of `map`: fetching a property.
    _.pluck = function(obj, key) {
        return _.map(obj, function(value) {
            return value[key];
        });
    };

    // Convenience version of a common use case of `filter`: selecting only objects
    // with specific `key:value` pairs.
    _.where = function(obj, attrs) {
        if (_.isEmpty(attrs)) 
            return [];
        return _.filter(obj, function(value) {
            for (var key in attrs) {
                if (attrs[key] !== value[key]) 
                    return false;
            }
            return true;
        });
    };

    // Return the maximum element or (element-based computation).
    // Can't optimize arrays of integers longer than 65,535 elements.
    // See: https://bugs.webkit.org/show_bug.cgi?id=80797
    _.max = function(obj, iterator, context) {
        if (!iterator && _.isArray(obj) && obj[0] === + obj[0] && obj.length < 65535) {
            return Math.max.apply(Math, obj);
        }
        if (!iterator && _.isEmpty(obj)) 
            return - Infinity;
        var result = {
            computed : - Infinity
        };
        each(obj, function(value, index, list) {
            var computed = iterator ? iterator.call(context, value, index, list) : value;
            computed >= result.computed && (result = {
                value : value,
                computed : computed
            });
        });
        return result.value;
    };

    // Return the minimum element (or element-based computation).
    _.min = function(obj, iterator, context) {
        if (!iterator && _.isArray(obj) && obj[0] === + obj[0] && obj.length < 65535) {
            return Math.min.apply(Math, obj);
        }
        if (!iterator && _.isEmpty(obj)) 
            return Infinity;
        var result = {
            computed : Infinity
        };
        each(obj, function(value, index, list) {
            var computed = iterator ? iterator.call(context, value, index, list) : value;
            computed < result.computed && (result = {
                value : value,
                computed : computed
            });
        });
        return result.value;
    };

    // Shuffle an array.
    _.shuffle = function(obj) {
        var rand;
        var index = 0;
        var shuffled = [];
        each(obj, function(value) {
            rand = _.random(index++);
            shuffled[index - 1] = shuffled[rand];
            shuffled[rand] = value;
        });
        return shuffled;
    };

    // An internal function to generate lookup iterators.
    var lookupIterator = function(value) {
        return _.isFunction(value) ? value : function(obj) {
            return obj[value];
        };
    };

    // Sort the object's values by a criterion produced by an iterator.
    _.sortBy = function(obj, value, context) {
        var iterator = lookupIterator(value);
        return _.pluck(_.map(obj, function(value, index, list) {
            return {
                value : value,
                index : index,
                criteria : iterator.call(context, value, index, list)
            };
        }).sort(function(left, right) {
            var a = left.criteria;
            var b = right.criteria;
            if (a !== b) {
                if (a > b || a === void 0) 
                    return 1;
                if (a < b || b === void 0) 
                    return - 1;
            }
            return left.index < right.index ? - 1 : 1;
        }), 'value');
    };

    // An internal function used for aggregate "group by" operations.
    var group = function(obj, value, context, behavior) {
        var result = {};
        var iterator = lookupIterator(value);
        each(obj, function(value, index) {
            var key = iterator.call(context, value, index, obj);
            behavior(result, key, value);
        });
        return result;
    };

    // Groups the object's values by a criterion. Pass either a string attribute
    // to group by, or a function that returns the criterion.
    _.groupBy = function(obj, value, context) {
        return group(obj, value, context, function(result, key, value) {
            (_.has(result, key) ? result[key] : (result[key] = [])).push(value);
        });
    };

    // Counts instances of an object that group by a certain criterion. Pass
    // either a string attribute to count by, or a function that returns the
    // criterion.
    _.countBy = function(obj, value, context) {
        return group(obj, value, context, function(result, key, value) {
            if (!_.has(result, key)) 
                result[key] = 0;
            result[key]++;
        });
    };

    // Use a comparator function to figure out the smallest index at which
    // an object should be inserted so as to maintain order. Uses binary search.
    _.sortedIndex = function(array, obj, iterator, context) {
        iterator = iterator == null ? _.identity : lookupIterator(iterator);
        var value = iterator.call(context, obj);
        var low = 0, high = array.length;
        while (low < high) {
            var mid = (low + high) >>> 1;
            iterator.call(context, array[mid]) < value ? low = mid + 1 : high = mid;
        }
        return low;
    };

    // Safely convert anything iterable into a real, live array.
    _.toArray = function(obj) {
        if (!obj) 
            return [];
        if (obj.length === + obj.length) 
            return slice.call(obj);
        return _.values(obj);
    };

    // Return the number of elements in an object.
    _.size = function(obj) {
        return (obj.length === + obj.length) ? obj.length : _.keys(obj).length;
    };

    // Array Functions
    // ---------------

    // Get the first element of an array. Passing **n** will return the first N
    // values in the array. Aliased as `head` and `take`. The **guard** check
    // allows it to work with `_.map`.
    _.first = _.head = _.take = function(array, n, guard) {
        return (n != null) && !guard ? slice.call(array, 0, n) : array[0];
    };

    // Returns everything but the last entry of the array. Especially useful on
    // the arguments object. Passing **n** will return all the values in
    // the array, excluding the last N. The **guard** check allows it to work with
    // `_.map`.
    _.initial = function(array, n, guard) {
        return slice.call(array, 0, array.length - ((n == null) || guard ? 1 : n));
    };

    // Get the last element of an array. Passing **n** will return the last N
    // values in the array. The **guard** check allows it to work with `_.map`.
    _.last = function(array, n, guard) {
        if ((n != null) && !guard) {
            return slice.call(array, Math.max(array.length - n, 0));
        } else {
            return array[array.length - 1];
        }
    };

    // Returns everything but the first entry of the array. Aliased as `tail` and `drop`.
    // Especially useful on the arguments object. Passing an **n** will return
    // the rest N values in the array. The **guard**
    // check allows it to work with `_.map`.
    _.rest = _.tail = _.drop = function(array, n, guard) {
        return slice.call(array, (n == null) || guard ? 1 : n);
    };

    // Trim out all falsy values from an array.
    _.compact = function(array) {
        return _.filter(array, function(value) {
            return !!value;
        });
    };

    // Internal implementation of a recursive `flatten` function.
    var flatten = function(input, shallow, output) {
        each(input, function(value) {
            if (_.isArray(value)) {
                shallow ? push.apply(output, value) : flatten(value, shallow, output);
            } else {
                output.push(value);
            }
        });
        return output;
    };

    // Return a completely flattened version of an array.
    _.flatten = function(array, shallow) {
        return flatten(array, shallow, []);
    };

    // Return a version of the array that does not contain the specified value(s).
    _.without = function(array) {
        return _.difference(array, slice.call(arguments, 1));
    };

    // Produce a duplicate-free version of the array. If the array has already
    // been sorted, you have the option of using a faster algorithm.
    // Aliased as `unique`.
    _.uniq = _.unique = function(array, isSorted, iterator, context) {
        var initial = iterator ? _.map(array, iterator, context) : array;
        var results = [];
        var seen = [];
        each(initial, function(value, index) {
            if (isSorted ? (!index || seen[seen.length - 1] !== value) : !_.contains(seen, value)) {
                seen.push(value);
                results.push(array[index]);
            }
        });
        return results;
    };

    // Produce an array that contains the union: each distinct element from all of
    // the passed-in arrays.
    _.union = function() {
        return _.uniq(concat.apply(ArrayProto, arguments));
    };

    // Produce an array that contains every item shared between all the
    // passed-in arrays.
    _.intersection = function(array) {
        var rest = slice.call(arguments, 1);
        return _.filter(_.uniq(array), function(item) {
            return _.every(rest, function(other) {
                return _.indexOf(other, item) >= 0;
            });
        });
    };

    // Take the difference between one array and a number of other arrays.
    // Only the elements present in just the first array will remain.
    _.difference = function(array) {
        var rest = concat.apply(ArrayProto, slice.call(arguments, 1));
        return _.filter(array, function(value) {
            return !_.contains(rest, value);
        });
    };

    // Zip together multiple lists into a single array -- elements that share
    // an index go together.
    _.zip = function() {
        var args = slice.call(arguments);
        var length = _.max(_.pluck(args, 'length'));
        var results = new Array(length);
        for (var i = 0; i < length; i++) {
            results[i] = _.pluck(args, "" + i);
        }
        return results;
    };

    // Converts lists into objects. Pass either a single array of `[key, value]`
    // pairs, or two parallel arrays of the same length -- one of keys, and one of
    // the corresponding values.
    _.object = function(list, values) {
        var result = {};
        for (var i = 0, l = list.length; i < l; i++) {
            if (values) {
                result[list[i]] = values[i];
            } else {
                result[list[i][0]] = list[i][1];
            }
        }
        return result;
    };

    // If the browser doesn't supply us with indexOf (I'm looking at you, **MSIE**),
    // we need this function. Return the position of the first occurrence of an
    // item in an array, or -1 if the item is not included in the array.
    // Delegates to **ECMAScript 5**'s native `indexOf` if available.
    // If the array is large and already in sort order, pass `true`
    // for **isSorted** to use binary search.
    _.indexOf = function(array, item, isSorted) {
        if (array == null) 
            return - 1;
        var i = 0, l = array.length;
        if (isSorted) {
            if (typeof isSorted == 'number') {
                i = (isSorted < 0 ? Math.max(0, l + isSorted) : isSorted);
            } else {
                i = _.sortedIndex(array, item);
                return array[i] === item ? i : - 1;
            }
        }
        if (nativeIndexOf && array.indexOf === nativeIndexOf) 
            return array.indexOf(item, isSorted);
        for (; i < l; i++) 
            if (array[i] === item) 
                return i;
        return - 1;
    };

    // Delegates to **ECMAScript 5**'s native `lastIndexOf` if available.
    _.lastIndexOf = function(array, item, from) {
        if (array == null) 
            return - 1;
        var hasIndex = from != null;
        if (nativeLastIndexOf && array.lastIndexOf === nativeLastIndexOf) {
            return hasIndex ? array.lastIndexOf(item, from) : array.lastIndexOf(item);
        }
        var i = (hasIndex ? from : array.length);
        while (i--) 
            if (array[i] === item) 
                return i;
        return - 1;
    };

    // Generate an integer Array containing an arithmetic progression. A port of
    // the native Python `range()` function. See
    // [the Python documentation](http://docs.python.org/library/functions.html#range).
    _.range = function(start, stop, step) {
        if (arguments.length <= 1) {
            stop = start || 0;
            start = 0;
        }
        step = arguments[2] || 1;

        var len = Math.max(Math.ceil((stop - start) / step), 0);
        var idx = 0;
        var range = new Array(len);

        while (idx < len) {
            range[idx++] = start;
            start += step;
        }

        return range;
    };

    // Function (ahem) Functions
    // ------------------

    // Reusable constructor function for prototype setting.
    var ctor = function() {};

    // Create a function bound to a given object (assigning `this`, and arguments,
    // optionally). Binding with arguments is also known as `curry`.
    // Delegates to **ECMAScript 5**'s native `Function.bind` if available.
    // We check for `func.bind` first, to fail fast when `func` is undefined.
    _.bind = function bind(func, context) {
        var bound, args;
        if (func.bind === nativeBind && nativeBind) 
            return nativeBind.apply(func, slice.call(arguments, 1));
        if (!_.isFunction(func)) 
            throw new TypeError;
        args = slice.call(arguments, 2);
        return bound = function() {
            if (!(this instanceof bound)) 
                return func.apply(context, args.concat(slice.call(arguments)));
            ctor.prototype = func.prototype;
            var self = new ctor;
            var result = func.apply(self, args.concat(slice.call(arguments)));
            if (Object(result) === result) 
                return result;
            return self;
        };
    };

    // Bind all of an object's methods to that object. Useful for ensuring that
    // all callbacks defined on an object belong to it.
    _.bindAll = function(obj) {
        var funcs = slice.call(arguments, 1);
        if (funcs.length == 0) 
            funcs = _.functions(obj);
        each(funcs, function(f) {
            obj[f] = _.bind(obj[f], obj);
        });
        return obj;
    };

    // Memoize an expensive function by storing its results.
    _.memoize = function(func, hasher) {
        var memo = {};
        hasher || (hasher = _.identity);
        return function() {
            var key = hasher.apply(this, arguments);
            return _.has(memo, key) ? memo[key] : (memo[key] = func.apply(this, arguments));
        };
    };

    // Delays a function for the given number of milliseconds, and then calls
    // it with the arguments supplied.
    _.delay = function(func, wait) {
        var args = slice.call(arguments, 2);
        return setTimeout(function() {
            return func.apply(null, args);
        }, wait);
    };

    // Defers a function, scheduling it to run after the current call stack has
    // cleared.
    _.defer = function(func) {
        return _.delay.apply(_, [func, 1].concat(slice.call(arguments, 1)));
    };

    // Returns a function, that, when invoked, will only be triggered at most once
    // during a given window of time.
    _.throttle = function(func, wait) {
        var context, args, timeout, throttling, more, result;
        var whenDone = _.debounce(function() {
            more = throttling = false;
        }, wait);
        return function() {
            context = this;
            args = arguments;
            var later = function() {
                timeout = null;
                if (more) {
                    result = func.apply(context, args);
                }
                whenDone();
            };
            if (!timeout) 
                timeout = setTimeout(later, wait);
            if (throttling) {
                more = true;
            } else {
                throttling = true;
                result = func.apply(context, args);
            }
            whenDone();
            return result;
        };
    };

    // Returns a function, that, as long as it continues to be invoked, will not
    // be triggered. The function will be called after it stops being called for
    // N milliseconds. If `immediate` is passed, trigger the function on the
    // leading edge, instead of the trailing.
    _.debounce = function(func, wait, immediate) {
        var timeout, result;
        return function() {
            var context = this, args = arguments;
            var later = function() {
                timeout = null;
                if (!immediate) 
                    result = func.apply(context, args);
            };
            var callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) 
                result = func.apply(context, args);
            return result;
        };
    };

    // Returns a function that will be executed at most one time, no matter how
    // often you call it. Useful for lazy initialization.
    _.once = function(func) {
        var ran = false, memo;
        return function() {
            if (ran) 
                return memo;
            ran = true;
            memo = func.apply(this, arguments);
            func = null;
            return memo;
        };
    };

    // Returns the first function passed as an argument to the second,
    // allowing you to adjust arguments, run code before and after, and
    // conditionally execute the original function.
    _.wrap = function(func, wrapper) {
        return function() {
            var args = [func];
            push.apply(args, arguments);
            return wrapper.apply(this, args);
        };
    };

    // Returns a function that is the composition of a list of functions, each
    // consuming the return value of the function that follows.
    _.compose = function() {
        var funcs = arguments;
        return function() {
            var args = arguments;
            for (var i = funcs.length - 1; i >= 0; i--) {
                args = [funcs[i].apply(this, args)];
            }
            return args[0];
        };
    };

    // Returns a function that will only be executed after being called N times.
    _.after = function(times, func) {
        if (times <= 0) 
            return func();
        return function() {
            if (--times < 1) {
                return func.apply(this, arguments);
            }
        };
    };

    // Object Functions
    // ----------------

    // Retrieve the names of an object's properties.
    // Delegates to **ECMAScript 5**'s native `Object.keys`
    _.keys = nativeKeys || function(obj) {
        if (obj !== Object(obj)) 
            throw new TypeError('Invalid object');
        var keys = [];
        for (var key in obj) 
            if (_.has(obj, key)) 
                keys[keys.length] = key;
        return keys;
    };

    // Retrieve the values of an object's properties.
    _.values = function(obj) {
        var values = [];
        for (var key in obj) 
            if (_.has(obj, key)) 
                values.push(obj[key]);
        return values;
    };

    // Convert an object into a list of `[key, value]` pairs.
    _.pairs = function(obj) {
        var pairs = [];
        for (var key in obj) 
            if (_.has(obj, key)) 
                pairs.push([key, obj[key]]);
        return pairs;
    };

    // Invert the keys and values of an object. The values must be serializable.
    _.invert = function(obj) {
        var result = {};
        for (var key in obj) 
            if (_.has(obj, key)) 
                result[obj[key]] = key;
        return result;
    };

    // Return a sorted list of the function names available on the object.
    // Aliased as `methods`
    _.functions = _.methods = function(obj) {
        var names = [];
        for (var key in obj) {
            if (_.isFunction(obj[key])) 
                names.push(key);
        }
        return names.sort();
    };

    // Extend a given object with all the properties in passed-in object(s).
    _.extend = function(obj) {
        each(slice.call(arguments, 1), function(source) {
            for (var prop in source) {
                obj[prop] = source[prop];
            }
        });
        return obj;
    };

    // Return a copy of the object only containing the whitelisted properties.
    _.pick = function(obj) {
        var copy = {};
        var keys = concat.apply(ArrayProto, slice.call(arguments, 1));
        each(keys, function(key) {
            if (key in obj) 
                copy[key] = obj[key];
        });
        return copy;
    };

    // Return a copy of the object without the blacklisted properties.
    _.omit = function(obj) {
        var copy = {};
        var keys = concat.apply(ArrayProto, slice.call(arguments, 1));
        for (var key in obj) {
            if (!_.contains(keys, key)) 
                copy[key] = obj[key];
        }
        return copy;
    };

    // Fill in a given object with default properties.
    _.defaults = function(obj) {
        each(slice.call(arguments, 1), function(source) {
            for (var prop in source) {
                if (obj[prop] == null) 
                    obj[prop] = source[prop];
            }
        });
        return obj;
    };

    // Create a (shallow-cloned) duplicate of an object.
    _.clone = function(obj) {
        if (!_.isObject(obj)) 
            return obj;
        return _.isArray(obj) ? obj.slice() : _.extend({}, obj);
    };

    // Invokes interceptor with the obj, and then returns obj.
    // The primary purpose of this method is to "tap into" a method chain, in
    // order to perform operations on intermediate results within the chain.
    _.tap = function(obj, interceptor) {
        interceptor(obj);
        return obj;
    };

    // Internal recursive comparison function for `isEqual`.
    var eq = function(a, b, aStack, bStack) {
        // Identical objects are equal. `0 === -0`, but they aren't identical.
        // See the Harmony `egal` proposal: http://wiki.ecmascript.org/doku.php?id=harmony:egal.
        if (a === b) 
            return a !== 0 || 1 / a == 1 / b;
        // A strict comparison is necessary because `null == undefined`.
        if (a == null || b == null) 
            return a === b;
        // Unwrap any wrapped objects.
        if (a instanceof _) 
            a = a._wrapped;
        if (b instanceof _) 
            b = b._wrapped;
        // Compare `[[Class]]` names.
        var className = toString.call(a);
        if (className != toString.call(b)) 
            return false;
        switch (className) {
            // Strings, numbers, dates, and booleans are compared by value.
        case '[object String]':
            // Primitives and their corresponding object wrappers are equivalent; thus, `"5"` is
            // equivalent to `new String("5")`.
            return a == String(b);
        case '[object Number]':
            // `NaN`s are equivalent, but non-reflexive. An `egal` comparison is performed for
            // other numeric values.
            return a != + a ? b != + b : (a == 0 ? 1 / a == 1 / b : a == + b);
        case '[object Date]':
        case '[object Boolean]':
            // Coerce dates and booleans to numeric primitive values. Dates are compared by their
            // millisecond representations. Note that invalid dates with millisecond representations
            // of `NaN` are not equivalent.
            return + a == + b;
            // RegExps are compared by their source patterns and flags.
        case '[object RegExp]':
            return a.source == b.source &&
            a.global == b.global &&
            a.multiline == b.multiline &&
            a.ignoreCase == b.ignoreCase;
        }
        if (typeof a != 'object' || typeof b != 'object') 
            return false;
        // Assume equality for cyclic structures. The algorithm for detecting cyclic
        // structures is adapted from ES 5.1 section 15.12.3, abstract operation `JO`.
        var length = aStack.length;
        while (length--) {
            // Linear search. Performance is inversely proportional to the number of
            // unique nested structures.
            if (aStack[length] == a) 
                return bStack[length] == b;
        }
        // Add the first object to the stack of traversed objects.
        aStack.push(a);
        bStack.push(b);
        var size = 0, result = true;
        // Recursively compare objects and arrays.
        if (className == '[object Array]') {
            // Compare array lengths to determine if a deep comparison is necessary.
            size = a.length;
            result = size == b.length;
            if (result) {
                // Deep compare the contents, ignoring non-numeric properties.
                while (size--) {
                    if (!(result = eq(a[size], b[size], aStack, bStack))) 
                        break;
                }
            }
        } else {
            // Objects with different constructors are not equivalent, but `Object`s
            // from different frames are.
            var aCtor = a.constructor, bCtor = b.constructor;
            if (aCtor !== bCtor && !(_.isFunction(aCtor) && (aCtor instanceof aCtor) &&
            _.isFunction(bCtor) && (bCtor instanceof bCtor))) {
                return false;
            }
            // Deep compare objects.
            for (var key in a) {
                if (_.has(a, key)) {
                    // Count the expected number of properties.
                    size++;
                    // Deep compare each member.
                    if (!(result = _.has(b, key) && eq(a[key], b[key], aStack, bStack))) 
                        break;
                }
            }
            // Ensure that both objects contain the same number of properties.
            if (result) {
                for (key in b) {
                    if (_.has(b, key) && !(size--)) 
                        break;
                }
                result = !size;
            }
        }
        // Remove the first object from the stack of traversed objects.
        aStack.pop();
        bStack.pop();
        return result;
    };

    // Perform a deep comparison to check if two objects are equal.
    _.isEqual = function(a, b) {
        return eq(a, b, [], []);
    };

    // Is a given array, string, or object empty?
    // An "empty" object has no enumerable own-properties.
    _.isEmpty = function(obj) {
        if (obj == null) 
            return true;
        if (_.isArray(obj) || _.isString(obj)) 
            return obj.length === 0;
        for (var key in obj) 
            if (_.has(obj, key)) 
                return false;
        return true;
    };

    // Is a given value a DOM element?
    _.isElement = function(obj) {
        return !!(obj && obj.nodeType === 1);
    };

    // Is a given value an array?
    // Delegates to ECMA5's native Array.isArray
    _.isArray = nativeIsArray || function(obj) {
        return toString.call(obj) == '[object Array]';
    };

    // Is a given variable an object?
    _.isObject = function(obj) {
        return obj === Object(obj);
    };

    // Add some isType methods: isArguments, isFunction, isString, isNumber, isDate, isRegExp.
    each(['Arguments', 'Function', 'String', 'Number', 'Date', 'RegExp'], function(name) {
        _['is' + name] = function(obj) {
            return toString.call(obj) == '[object ' + name + ']';
        };
    });

    // Define a fallback version of the method in browsers (ahem, IE), where
    // there isn't any inspectable "Arguments" type.
    if (!_.isArguments(arguments)) {
        _.isArguments = function(obj) {
            return !!(obj && _.has(obj, 'callee'));
        };
    }

    // Optimize `isFunction` if appropriate.
    if (typeof (/./) !== 'function') {
        _.isFunction = function(obj) {
            return typeof obj === 'function';
        };
    }

    // Is a given object a finite number?
    _.isFinite = function(obj) {
        return _.isNumber(obj) && isFinite(obj);
    };

    // Is the given value `NaN`? (NaN is the only number which does not equal itself).
    _.isNaN = function(obj) {
        return _.isNumber(obj) && obj != + obj;
    };

    // Is a given value a boolean?
    _.isBoolean = function(obj) {
        return obj === true || obj === false || toString.call(obj) == '[object Boolean]';
    };

    // Is a given value equal to null?
    _.isNull = function(obj) {
        return obj === null;
    };

    // Is a given variable undefined?
    _.isUndefined = function(obj) {
        return obj === void 0;
    };

    // Shortcut function for checking if an object has a given property directly
    // on itself (in other words, not on a prototype).
    _.has = function(obj, key) {
        return hasOwnProperty.call(obj, key);
    };

    // Utility Functions
    // -----------------

    // Run Underscore.js in *noConflict* mode, returning the `_` variable to its
    // previous owner. Returns a reference to the Underscore object.
    _.noConflict = function() {
        root._ = previousUnderscore;
        return this;
    };

    // Keep the identity function around for default iterators.
    _.identity = function(value) {
        return value;
    };

    // Run a function **n** times.
    _.times = function(n, iterator, context) {
        for (var i = 0; i < n; i++) 
            iterator.call(context, i);
    };

    // Return a random integer between min and max (inclusive).
    _.random = function(min, max) {
        if (max == null) {
            max = min;
            min = 0;
        }
        return min + (0 | Math.random() * (max - min + 1));
    };

    // List of HTML entities for escaping.
    var entityMap = {
        escape: {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;'
        }
    };
    entityMap.unescape = _.invert(entityMap.escape);

    // Regexes containing the keys and values listed immediately above.
    var entityRegexes = {
        escape: new RegExp('[' + _.keys(entityMap.escape).join('') + ']', 'g'),
        unescape: new RegExp('(' + _.keys(entityMap.unescape).join('|') + ')', 'g')
    };

    // Functions for escaping and unescaping strings to/from HTML interpolation.
    _.each(['escape', 'unescape'], function(method) {
        _[method] = function(string) {
            if (string == null) 
                return '';
            return ('' + string).replace(entityRegexes[method], function(match) {
                return entityMap[method][match];
            });
        };
    });

    // If the value of the named property is a function then invoke it;
    // otherwise, return it.
    _.result = function(object, property) {
        if (object == null) 
            return null;
        var value = object[property];
        return _.isFunction(value) ? value.call(object) : value;
    };

    // Add your own custom functions to the Underscore object.
    _.mixin = function(obj) {
        each(_.functions(obj), function(name) {
            var func = _[name] = obj[name];
            _.prototype[name] = function() {
                var args = [this._wrapped];
                push.apply(args, arguments);
                return result.call(this, func.apply(_, args));
            };
        });
    };

    // Generate a unique integer id (unique within the entire client session).
    // Useful for temporary DOM ids.
    var idCounter = 0;
    _.uniqueId = function(prefix) {
        var id = idCounter++;
        return prefix ? prefix + id : id;
    };

    // By default, Underscore uses ERB-style template delimiters, change the
    // following template settings to use alternative delimiters.
    _.templateSettings = {
        evaluate : /<%([\s\S]+?)%>/g,
        interpolate : /<%=([\s\S]+?)%>/g,
        escape : /<%-([\s\S]+?)%>/g
    };

    // When customizing `templateSettings`, if you don't want to define an
    // interpolation, evaluation or escaping regex, we need one that is
    // guaranteed not to match.
    var noMatch = /(.)^/;

    // Certain characters need to be escaped so that they can be put into a
    // string literal.
    var escapes = {
        "'": "'",
        '\\': '\\',
        '\r': 'r',
        '\n': 'n',
        '\t': 't',
        '\u2028': 'u2028',
        '\u2029': 'u2029'
    };

    var escaper = /\\|'|\r|\n|\t|\u2028|\u2029/g;

    // JavaScript micro-templating, similar to John Resig's implementation.
    // Underscore templating handles arbitrary delimiters, preserves whitespace,
    // and correctly escapes quotes within interpolated code.
    _.template = function(text, data, settings) {
        settings = _.defaults({}, settings, _.templateSettings);

        // Combine delimiters into one regular expression via alternation.
        var matcher = new RegExp([
        (settings.escape || noMatch).source,
        (settings.interpolate || noMatch).source,
        (settings.evaluate || noMatch).source
        ].join('|') + '|$', 'g');

        // Compile the template source, escaping string literals appropriately.
        var index = 0;
        var source = "__p+='";
        text.replace(matcher, function(match, escape, interpolate, evaluate, offset) {
            source += text.slice(index, offset)
            .replace(escaper, function(match) {
                return '\\' + escapes[match];
            });
            source +=
            escape ? "'+\n((__t=(" + escape + "))==null?'':_.escape(__t))+\n'" :
            interpolate ? "'+\n((__t=(" + interpolate + "))==null?'':__t)+\n'" :
            evaluate ? "';\n" + evaluate + "\n__p+='" : '';
            index = offset + match.length;
        });
        source += "';\n";

        // If a variable is not specified, place data values in local scope.
        if (!settings.variable) 
            source = 'with(obj||{}){\n' + source + '}\n';

        source = "var __t,__p='',__j=Array.prototype.join," +
        "print=function(){__p+=__j.call(arguments,'');};\n" +
        source + "return __p;\n";

        try {
            var render = new Function(settings.variable || 'obj', '_', source);
        } catch (e) {
            e.source = source;
            throw e;
        }

        if (data) 
            return render(data, _);
        var template = function(data) {
            return render.call(this, data, _);
        };

        // Provide the compiled function source as a convenience for precompilation.
        template.source = 'function(' + (settings.variable || 'obj') + '){\n' + source + '}';

        return template;
    };

    // Add a "chain" function, which will delegate to the wrapper.
    _.chain = function(obj) {
        return _(obj).chain();
    };

    // OOP
    // ---------------
    // If Underscore is called as a function, it returns a wrapped object that
    // can be used OO-style. This wrapper holds altered versions of all the
    // underscore functions. Wrapped objects may be chained.

    // Helper function to continue chaining intermediate results.
    var result = function(obj) {
        return this._chain ? _(obj).chain() : obj;
    };

    // Add all of the Underscore functions to the wrapper object.
    _.mixin(_);

    // Add all mutator Array functions to the wrapper.
    each(['pop', 'push', 'reverse', 'shift', 'sort', 'splice', 'unshift'], function(name) {
        var method = ArrayProto[name];
        _.prototype[name] = function() {
            var obj = this._wrapped;
            method.apply(obj, arguments);
            if ((name == 'shift' || name == 'splice') && obj.length === 0) 
                delete obj[0];
            return result.call(this, obj);
        };
    });

    // Add all accessor Array functions to the wrapper.
    each(['concat', 'join', 'slice'], function(name) {
        var method = ArrayProto[name];
        _.prototype[name] = function() {
            return result.call(this, method.apply(this._wrapped, arguments));
        };
    });

    _.extend(_.prototype, {

        // Start chaining a wrapped Underscore object.
        chain: function() {
            this._chain = true;
            return this;
        },

        // Extracts the result from a wrapped and chained object.
        value: function() {
            return this._wrapped;
        }

    });

    return _.noConflict();
});

define('nodeDataTooltip', [
'jquery',
'underscore',
'internal/common/tooltip'],
function nodeDataTooltip($, _) {

    var existingTooltips = [],
    tooltipCount = 0,
    $tt,
    defaults = {
        content: "",
        style: "mini"
    };

    // Merge the default options with any data-tooltip-* options specified in markup
    var parseOptions = function($el) {
        var data = _.reduce($el.parents(), function(el, par) {
            return _.extend(el, $(par).data());
        }, $el.data());
        var opts = _.reduce(_.keys(data), function(opts, key) {
            if (key.substr(0, 7) === "tooltip") {
                opts[key.substr(7).toLowerCase()] = data[key];
            }
            return opts;
        }, _.clone(defaults));

        if (opts.offsetleft || opts.offsettop) {
            opts.offset = [parseInt(opts.offsetleft, 10), parseInt(opts.offsettop, 10)];
        }

        if (opts.anchor === "this") {
            opts.anchor = $el;
        }

        return opts;
    };

    var showTooltipFor = function($el) {
        hideTooltip();
        if (!$el.data('existingTooltip')) {
            $tt = $.tooltip(parseOptions($el));
            $tt.prepend($el.data("tooltip"));
            if ($el.data('tooltip-title')) {
                $tt.prepend("<h6>" + $el.data('tooltip-title') + "</h6>");
            }
            existingTooltips[tooltipCount] = $tt;
            $el.data({
                existingTooltip: tooltipCount
            });
            tooltipCount++;
        } else {
            $tt = existingTooltips[$el.data('existingTooltip')];
        }
        $tt.addClass("active");
        $tt.show();
    };

    var hideTooltip = function() {
        if ($tt) {
            $tt.removeClass("active");
            $tt.hide();
        }
    };

    $(function() {
        $("body").on("mouseover", "*[data-tooltip]", function() {
            showTooltipFor($(this));
        }).on("mouseout", "*[data-tooltip]", function() {
            hideTooltip();
        });

        $("body").on("showTooltip", function(e, el) {
            showTooltipFor($(el));
        }).on("hideTooltip", function(e, el) {
            hideTooltip();
        });
    });

});

/**
* These classes are used both in the editor and on the user's site
* In edit mode, functionality is added to these classes in bldr.modules.js
* Therefore, this file should only contain SHARED methods
*/

define('internal/sitebuilder/common/webs.modules', [
'require',
'jquery',
'internal/sitebuilder/common/log',
'spine',
'internal/common/tooltip',
'nodeDataTooltip'
], function(require, $, log, Spine) {
    var Module = Spine.Controller.create({
        init: function(options) {
            this.data = options.data;
        },

        oneLoaded: function() {
            // modules should override this
        },

        // We're overriding Spine.Event.trigger so that buggy callbacks don't mess up the call stack
        trigger: function() {
            var args = Spine.makeArray(arguments);
            var ev = args.shift();

            var list, calls, i, l;
            if (!(calls = this.hasOwnProperty("_callbacks") && this._callbacks)) 
                return false;
            if (!(list = this._callbacks[ev])) 
                return false;

            for (i = 0, l = list.length; i < l; i++) {
                try {
                    if (list[i].apply(this, args) === false)
                        return false;
                } catch (error) {
                    log.trigger("Modules", "error", "Module:" + this.parent.slug + ", Event:" + ev, error);
                }
            }

            return true;
        }
    }),
    CustomModule = Module.create(),
    CompositeModule = CustomModule.create(),
    WidgetModule = Module.create(),
    IframeModule = Module.create(),
    AppModule = Module.create();

    // Class Methods
    Module.extend({
        styles: []
    });

    CompositeModule.include({
        /**
        			 * Subclasses of CompositeModule SHOULD override this method if there is
        			 * any submodule *SLUG* where data.*SLUG* is not the module data for the submodule,
        			 * it should return an array of objects describing each submodule, e.g.:
        			 * [
        			 *   {
        			 *     name: "unique identifier (within the scope of this composite module) for the submodule",
        			 *     el: "root container for the submodule",
        			 *     slug: "name of module class",
        			 *     data: {data: "for", the: "submodule"},
        			 *   },
        			 *   {
        			 *     ...
        			 *   },
        			 * ]
        			 */
        describeSubmodules: function() {
            var self = this,
            $top = self.el.children(".webs-composite-module").eq(0),
            submoduleContainers = $top.find(".webs-submodule"),
            submoduleDescriptions = [];
            $.each(submoduleContainers, function(index) {
                var container = submoduleContainers.eq(index);
                if (container.parents(".webs-composite-module")[0] === $top[0]) {
                    var slug = container.attr("webs-submodule-slug");
                    submoduleDescriptions.push({
                        name: slug,
                        el: container,
                        slug: container.attr("webs-submodule-slug"),
                        data: self.data[slug]
                    });
                }
            });
            return submoduleDescriptions;
        },


        bindSubmodules: function() {
            var self = this,
            descriptions = this.describeSubmodules(),
            loadedDeferreds = [];

            self.submoduleInstances = {};
            require(['internal/sitebuilder/common/ModuleClassLoader'], function(ModuleClassLoader) {
                $.each(descriptions, function(i) {
                    var desc = descriptions[i];
                    var deferred = new $.Deferred();
                    loadedDeferreds.push(deferred);
                    ModuleClassLoader.create(desc.slug, {
                        el: desc.el,
                        data: desc.data
                    }).done(function(submodule) {
                        self.submoduleInstances[desc.name] = submodule;
                        desc.el.data("controller", submodule);
                        deferred.resolve();
                    });
                });
            });
            return $.when.apply($, loadedDeferreds);
        },

        oneLoaded: function() {
            this.bindSubmodules().done($.proxy(function() {
                $.each(this.submoduleInstances, function(name, submodule) {
                    submodule.oneLoaded();
                });
            }, this));
        }
    });

    webs.modules = {
        Module: Module,
        CustomModule: CustomModule,
        IframeModule: IframeModule,
        WidgetModule: WidgetModule,
        AppModule: AppModule,
        CompositeModule: CompositeModule
    };

    return webs.modules;
});

define('internal/sitebuilder/common/ModuleClassLoader', [
'jquery',
'moduleVersions',
'internal/sitebuilder/common/log',
'spine',
'internal/sitebuilder/common/webs.modules',
'internal/sitebuilder/common/creativeCommons'
], function($, moduleVersions, log, Spine, webs_modules) {
    var MODULE_LOAD_TIMEOUT = 30000;
    var RESOLVED_PROMISE = $.Deferred().resolve().promise();

    var ModuleClassLoader = Spine.Class.create({
        init: function(options) {
            this.MODULES_URL = options.MODULES_URL;
            this.MODULES_VERSION = options.MODULES_VERSION;

            /**
            			* Map of module slugs to classes
            			*/
            this.classes = {};

            /**
            			* Map from module slugs to a promises for when it loads
            			*/
            this.modulePromises = {};

            /**
            			 * Map from module slugs to promises for when the JS file loads
            			 */
            this.moduleJSPromises = {};

        },

        getClass: function(moduleSlug) {
            return this.classes[moduleSlug];
        },

        /**
        		* Called from the module definition file. Each module type is only registered once.
        		*/
        register: function(moduleSlug, include, extend) {
            var superClass;
            extend.slug = moduleSlug;

            if (extend.iframe) {
                superClass = webs_modules.IframeModule;
            } else if (extend.isWidget) {
                superClass = webs_modules.WidgetModule;
            } else if (extend.submodules) {
                superClass = webs_modules.CompositeModule;
            } else {
                superClass = webs_modules.CustomModule;
            }

            this.classes[moduleSlug] = superClass.create(include, extend);
            if (!this.moduleJSPromises[moduleSlug]) {
                this.moduleJSPromises[moduleSlug] = new $.Deferred();
            }
            this.moduleJSPromises[moduleSlug].resolve();
        },

        /**
        		 * This create method is used when we KNOW that we have already loaded the module
        		 * @param moduleClass
        		 * @param options
        		 * @param deferred
        		 * @returns {*}
        		 * @private
        		 */
        _create: function(moduleClass, options, deferred) {
            var moduleSlug = moduleClass.slug,
            proto = $.extend({}, moduleClass),
            data = $.extend(true, {}, moduleClass.defaultData, options.data),
            style = moduleClass.styles[data._style] || moduleClass.defaultStyle;

            // TODO: HACK for buckets. jQuery deep extend also does arrays :(
            if (data.bucketContents && options.data && options.data.bucketContents) {
                data.bucketContents = options.data.bucketContents;
                // Hack for SITEBUILDER-2017 and SITEBUILDER-2025
                // Sometimes the image values were stored as NaN which caused image to come back as null
                // So we need to default it
                for (var i = 0, len = data.cols.length; i < len; i++) {
                    if (!data.cols[i].image) {
                        log.error("An image within buckets was null");
                        data.cols[i].image = moduleClass.defaultData.cols[0].image;
                    }
                }
            }
            if (moduleSlug == "contact_form" && options.data && options.data.fields) 
                data.fields = options.data.fields;

            var obj = this.getClassForStyle(moduleClass, style).init({
                el: options.container || options.el,
                data: data
            });

            deferred.resolve(obj);

            this.trigger('created', obj);

            return obj;
        },

        /**
        		 * Essentially the same as create, but for appModules.
        		 * Sidebar modules are different. We don't call "load" on them.
        		 */
        createAppModule: function(moduleSlug, options) {
            var moduleClass = this.getClass(moduleSlug),
            deferred = $.Deferred();

            if (!moduleClass) {
                // If module is not registered, register a fake one and log an error
                moduleClass = this.classes[moduleSlug] = webs_modules.AppModule.create({
                    id: moduleSlug.substring(12)
                });
                log.error("Unable to load appmodule", moduleSlug);
            }

            this._create(moduleClass, options, deferred);
            return deferred.promise();
        },

        create: function(moduleSlug, options) {
            if (moduleSlug.indexOf('app-sidebar-') === 0) 
                return this.createAppModule(moduleSlug, options);

            var self = this,
            moduleClass = this.getClass(moduleSlug),
            deferred = $.Deferred();

            self.load(moduleSlug).done(function() {
                self._create(self.getClass(moduleSlug), options, deferred);
            }).fail(deferred.reject);

            return deferred.promise();
        },

        /**
        		* Given a module class and a style, return the proper class to init
        		* Styles have their own subclasses of the moduleClass, because they can add methods
        		* TODO: This should probably move to Module.getClassForStyle
        		*/
        getClassForStyle: function(moduleClass, style) {
            if (style) {
                /* jshint ignore:start */
                // Collect all styles (because styles can inherit from other styles)
                var styles = [];
                do {
                    styles.push(style);
                }
                while (style = moduleClass.styles[style.inherit]);
                /* jshint ignore:end */

                // Iterate through the styles in reverse, so that the base style is first, and the specified style is last
                var superClass = moduleClass,
                subClass,
                s;
                for (var i = styles.length - 1; i >= 0; i--) {
                    s = styles[i];
                    subClass = s.klass;
                    if (!subClass) {
                        delete s.defaultStyle;

                        if (s.global.js) {
                            // Extend the super class
                            var include = {}, extend = {};
                            s.global.js(include, extend);
                            s.klass = subClass = superClass.create(include, extend);
                        } else {
                            // This happens when a style doesn't change the JS, but does change the CSS
                            // No JS to add. Just use the super class
                            s.klass = subClass = superClass;
                        }
                    }
                    superClass = subClass;
                }

                return superClass;
            } else {
                // How!?!?
                // TODO: In the future, not all modules will need a style
                if (log) 
                    log.trigger("Modules", "info", 'No style found for ' + moduleClass.slug + ' module!', moduleClass);
                return moduleClass;
            }
        },

        getModuleSlugFromUrl: function(url) {
            var urlParts = url.split("/");
            var moduleSlug;
            if (urlParts[2] == "modules") {
                moduleSlug = urlParts[3];
            } else {
                moduleSlug = urlParts[3] + "_" + urlParts[urlParts.length - 1].replace(".less", "");
            }
            return moduleSlug;
        },

        getModuleVersion: function(slug) {
            var version;
            if (moduleVersions && slug in moduleVersions) {
                version = "v" + moduleVersions[slug];
            } else {
                log.warn("WARNING: Retrieving unversioned asset for module " + slug);
                version = this.MODULES_VERSION;
            }
            return version;
        },

        getModuleAssetURL: function(slug, path) {
            return this.MODULES_URL + slug + '/' + this.getModuleVersion(slug) + '/' + path;
        },

        cssPath: function(slug) {
            return this.getModuleAssetURL(slug, 'view.packaged.less');
        },

        /**
        		* Load a module class from the backend
        		*/
        load: function(slug) {
            if (this.modulePromises[slug]) 
                return this.modulePromises[slug];

            var self = this,
            cssLoadedPromise = this.loadCss(this.cssPath(slug)),
            dependenciesLoadedPromise = $.Deferred();
            this.modulePromises[slug] = $.Deferred();

            if (!this.moduleJSPromises[slug]) {
                this.moduleJSPromises[slug] = new $.Deferred();
            }

            this.loadJs(slug);

            this.moduleJSPromises[slug].done(function() {
                self.loadModuleDependencies(slug).done(dependenciesLoadedPromise.resolve).fail(dependenciesLoadedPromise.reject);
            }).fail(dependenciesLoadedPromise.reject);

            // Module is loaded when css and js are all loaded.
            $.when(cssLoadedPromise, dependenciesLoadedPromise)
            .done(self.modulePromises[slug].resolve)
            .fail(self.modulePromises[slug].reject);

            // Trigger error callbacks if not laoded quickly.
            setTimeout(this.modulePromises[slug].reject, MODULE_LOAD_TIMEOUT);

            return this.modulePromises[slug];
        },

        loadAll: function(types) {
            var deferredBulkLoad = $.Deferred(),
            typePromises = [];
            for (var i = 0, len = types.length; i < len; i++) {
                typePromises.push(this.load(types[i]));
            }
            $.when.apply($, typePromises).done(deferredBulkLoad.resolve).fail(deferredBulkLoad.reject);
            return deferredBulkLoad.promise();
        },

        loadCss: function(url) {
            // In view mode, all the CSS is already loaded
            return RESOLVED_PROMISE;
        },

        loadJs: function(moduleSlug) {
            var deferred = $.Deferred();
            require([this.getModuleAssetURL(moduleSlug, moduleSlug + '_view.js')], null);
            return deferred;
        },

        /* Loads theme style css and returns a list of theme style js files */
        themeStyleFileList: function(moduleSlug) {
            // Per-Theme Module Styles
            var self = this,
            moduleClass = this.getClass(moduleSlug),
            theme = webs.theme,
            files = [],
            cssPromises = [];
            if (theme.moduleStyles && theme.moduleStyles[moduleSlug]) {
                $.each(theme.moduleStyles[moduleSlug], function(styleSlug, def) {
                    if (log) 
                        log.trigger("Modules", "debug", 'Loading theme style ' + styleSlug + ' for ' + moduleSlug);
                    moduleClass.styles[styleSlug] = def;
                    def.slug = styleSlug;
                    if (def.global.js) 
                        files.push(theme.url + "/" + def.global.js);
                    if (def.global.css) 
                        cssPromises.push(self.loadCss(theme.url + '/' + def.global.css));

                    // Set this as the default style
                    if (def['default']) 
                        moduleClass.defaultStyle = def;
                });
            }
            return [files, $.when.apply($, cssPromises)];
        },

        /* A list of module names the given module depends on */
        moduleDependencyList: function(moduleSlug) {
            var submodules = [],
            moduleClass = this.getClass(moduleSlug);

            // Module js dependencies
            if (moduleClass.submodules) {
                $.each(moduleClass.submodules, function(slug, sm) {
                    submodules.push(slug);
                });
            }

            return submodules;
        },

        shouldConsolidateAssets: function(moduleSlug) {
            return moduleSlug.indexOf("zumba") === - 1 && moduleSlug.indexOf("app-sidebar") === - 1;
        },

        /* Loads both theme styles and module dependencies for the given module. */
        loadModuleDependencies: function(moduleSlug) {
            var self = this,
            dependencies = this.themeStyleFileList(moduleSlug),
            jsFiles = dependencies[0],
            cssPromise = dependencies[1],
            modules = this.moduleDependencyList(moduleSlug),
            total = jsFiles.length + modules.length,
            loaded = 0;

            if (total === 0) {
                return cssPromise;
            } else {
                var deferred = $.Deferred(),
                allRequiredPromises = [cssPromise];
                $.each(modules, function(index, moduleSlug) {
                    allRequiredPromises.push(self.load(moduleSlug));
                });
                $.each(jsFiles, function(index, file) {
                    var jsPromise = $.Deferred();
                    allRequiredPromises.push(jsPromise);
                    require([script(file)], jsPromise.resolve);
                });
                $.when.apply($, allRequiredPromises).done(deferred.resolve).fail(deferred.reject);
                return deferred.promise();
            }
        }

    });
    ModuleClassLoader.include(Spine.Events);

    var mcl = ModuleClassLoader.init({
        MODULES_URL: webs.props.dynamicAssetServer + '/s/modules/',
        MODULES_VERSION: webs.props.modulesVersion
    });

    var deferredModules = $.Deferred();
    mcl.renderedModulesPromise = deferredModules.promise();
    $(function() {
        $.when.apply($, webs.renderedModulesPromises || []).done(deferredModules.resolve).fail(deferredModules.reject);
    });

    webs_modules.ModuleClassLoader = mcl;

    return mcl;
});


(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define('jquery.watch', ['jquery'], factory);
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function ($) {
    // A jQuery module for watching changes to the attributes of DOM nodes.

    // TODO: alternate implementation strategies, using DOMAttrModified in FF, propertychange in IE.

    $.fn.watch = function( attr, callback ) {
        this.each(function() {
            var self = this,
            $this = $(this),
            val = $this.prop(attr);

            $this.data(attr + ".watch", setInterval(
            function() {
                var newVal = $this.prop(attr);
                if (val != newVal) {
                    callback({
                        target: this,
                        attribute: attr,
                        newValue: newVal,
                        oldValue: val
                    });
                    val = newVal;
                }
            }, 100));
        });

        return this;
    };

    $.fn.unwatch = function( attr ) {
        this.each(function() {
            var key = attr + ".watch";
            clearInterval($(this).data(key));
            $(this).removeData(key);
        });
    };
}));
define('jquery.customRange', [
'jquery',
'jquery.watch'
], function($) {

    /*  LIGHTWEIGHT DRAG AND DROP

    		$(".myelement").drag({y: false}).bind("drag", function(event, x, y) {
    			// do your custom thing
    		});

    		Configuration:
    			x: true,        // enable horizontal drag
    			y: true,        // enable vertical drag
    			drag: true      // true = perform drag, false = only fire events

    		Events: dragStart, drag, dragEnd.

    		FIXME: This doesn't work with multitouch, assistive devices, etc.
    	*/
    var doc, draggable;

    $.fn.drag = function(conf) {

        // disable IE specialities
        document.ondragstart = function () {
            return false;
        };

        conf = $.extend({
            x: true,
            y: true,
            drag: true
        }, conf);

        doc = doc || $(document).bind("mousedown mouseup", function(e) {

            var el = $(e.target);

            // start
            if (e.type == "mousedown" && el.data("drag")) {

                var offset = el.position(),
                x0 = e.pageX - offset.left,
                y0 = e.pageY - offset.top,
                start = true;

                doc.bind("mousemove.drag", function(e) {
                    var x = e.pageX - x0 + 6,
                    y = e.pageY - y0,
                    props = {};

                    if (conf.x) {
                        props.left = x;
                    }
                    if (conf.y) {
                        props.top = y;
                    }

                    if (start) {
                        el.trigger("dragStart");
                        start = false;
                    }
                    if (conf.drag) {
                        el.css(props);
                    }
                    el.trigger("drag", [y, x]);
                    draggable = el;
                });

                e.preventDefault();

            } else {

                try {
                    if (draggable) {
                        draggable.trigger("dragEnd");
                    }
                } finally {
                    doc.unbind("mousemove.drag");
                    draggable = null;
                }
            }

        });

        return this.data("drag", true);
    };

    var rateLimit = function(f, ms) {
        var called = false;

        return function() {
            if (!called) {
                called = true;
                setTimeout(function() {
                    called = false;
                }, ms);
                f.apply(null, arguments);
            }
        };
    };

    var customrange = function(el, conf) {

        var step = function() {
            return + $input.attr('step') || 1;
        },
        min = function() {
            return + $input.attr('min') || 0;
        },
        max = function() {
            return + $input.attr('max') || 1;
        },
        range = function() {
            return max() - min();
        },
        minX = function() {
            return 0;
        },
        maxX = function() {
            return $slider.innerWidth();
        },
        width = function() {
            return maxX() - minX();
        },
        posToPct = function(pos) {
            return (pos - minX()) / width();
        },
        valToPct = function(val) {
            return (val - min()) / range();
        },
        pctToPos = function(pct) {
            return pct * width() + minX();
        },
        pctToVal = function(pct) {
            return pct * range() + min();
        },
        valToPos = function(val) {
            return pctToPos(valToPct(val));
        },
        posToVal = function(pos) {
            return pctToVal(posToPct(pos));
        },
        setVal = function(val) {
            val = normalize(val);
            $input.val(val).change();
            return val;
        },
        normalize = function(val) {
            return Math.min(Math.max(val, min()), max());
        },
        moveSlider = function(pos) {
            $progress.css({
                width: Math.max(0, parseInt(pos, 10)) || 0
            });
            var halfWidth = $handle.innerWidth() / 2,
            left = Math.max( - halfWidth, pos - halfWidth);
            $handle.css({
                left: left
            });
        },
        refresh = function() {
            moveSlider(valToPos($input.val()));
        },
        onDrag = function(e, y, x) {
            if ($input.is(":disabled")) {
                return false;
            }
            var val = posToVal(x);
            val = setVal(val);
            moveSlider(valToPos(val));
            return this;
        },
        // DOM elements
        $input = el.hide(),
        $slider = $('<div class="slider"/>').append(
        '<a class="zoom zoom-out">-</a>',
        '<div class="progress"/>',
        '<a href="#" class="handle"/>',
        '<a class="zoom zoom-in">+</a>'
        ),
        $handle = $slider.find(".handle"),
        $progress = $slider.find(".progress");

        // Add events
        $input.change(refresh);
        $input.watch("min", refresh);
        $input.watch("max", refresh);
        $input.watch("step", refresh);

        $slider.mousedown(function(e) {
            // FIXME cache this or something.
            var $this = $(this),
            cX = e.clientX,
            eOff = $this.offset().left,
            over = e.clientX > eOff + $this.width(),
            under = e.clientX < eOff,
            step = range() / 10,
            pos;

            if (over) {
                pos = valToPos(parseInt($input.val(), 10) + step);
            } else if (under) {
                pos = valToPos(parseInt($input.val(), 10) - step);
            } else {
                pos = cX - eOff + 3;
            }
            onDrag(e, 0, pos);
        });


        $handle.drag({
            drag: false
        }).
        bind("dragStart", function() {
            if (conf && typeof conf.dragStart === 'function') 
                conf.dragStart();
        }).
        bind("dragEnd", function() {
            if (conf && typeof conf.dragEnd === 'function') 
                conf.dragEnd();
        }).
        bind("drag", rateLimit(onDrag, 40)).
        click(function(e) {
            return e.preventDefault();
        });

        $input.before($slider);
        refresh();
    };

    $.expr[':'].range = function(el) {
        var type = el.getAttribute("type");
        return type && type == 'range' || !!$(el).filter("input").data("customrange");
    };

    $.fn.customrange = function(conf) {
        this.each(function() {
            if (!$.data(this, "customrange")) {
                $.data(this, "customrange", true);
                customrange($(this), conf);
            }
        });
        return this;
    };
});

require([
'jquery'
], function jquery_numberInput($) {

    var polyfillTemplate = '<div class="numberInput"><input type="text"><a class="increment"></a><a class="decrement"></a></div>';

    $.fn.numberInput = function numberInput(opts) {
        var instance = {
            $el: this,
            value: parseFloat(opts && opts.value || this.attr('value') || 0),
            min: parseFloat(opts && opts.min || this.attr('min') || - Infinity),
            max: parseFloat(opts && opts.max || this.attr('max') || Infinity),
            step: parseFloat(opts && opts.step || this.attr('step') || 1)
        },
        $numberInput = $(polyfillTemplate).
        width (this.outerWidth()).
        data ('numberInput', instance);
        instance.set = $.proxy(set, $numberInput);
        this.data("numberInput", $numberInput);

        this.before($numberInput).hide();
        update($numberInput, instance, true);

        $numberInput.
        on('click', '.numberInput>.increment', increment).
        on('click', '.numberInput>.decrement', decrement).
        on('keypress', '.numberInput>input', keypress).
        on('keydown', '.numberInput>input', keydown).
        on('change', '.numberInput>input', change).
        on('paste', '.numberInput>input', change).
        on('cut', '.numberInput>input', change);

        return $numberInput;
    };

    function set(value) {
        var instance = this.data('numberInput');
        instance.value = value;
        update(this, instance, true);
    }

    function increment(e) {
        var $numberInput = $(e.currentTarget).closest('.numberInput'),
        instance = $numberInput.data('numberInput');
        instance.value += instance.step;
        update($numberInput, instance);
    }

    function decrement(e) {
        var $numberInput = $(e.currentTarget).closest('.numberInput'),
        instance = $numberInput.data('numberInput');
        instance.value -= instance.step;
        update($numberInput, instance);
    }

    function keypress(e) {
        // Only allow certain characters to be inserted
        if ((e.charCode < 48 || 58 <= e.charCode) /* Digits */
        && e.charCode != 13 /* Enter (e.g. to submit) */
        && e.charCode != 45 /* - (negative) */
        && e.charCode != 46 /* . (floats) */
        ) {
            e.preventDefault();
            return false;
        }
        // Ok, let it be inserted, but validate immediately afterward.
        var $numberInput = $(e.currentTarget).closest('.numberInput'),
        instance = $numberInput.data('numberInput');
        instance.updateTimeout && clearTimeout(instance.updateTimeout)
        instance.updateTimeout = setTimeout(function() {
            delete instance.updateTimeout;
            instance.value = parseFloat($numberInput.find('input').val());
            update($numberInput, instance);
        }, 1);
    }

    function keydown(e) {
        // On delete or backspace, revalidate
        if (e.keyCode == 8 /*delete*/
        || e.keyCode == 46 /*backspace*/
        ) {
            var $numberInput = $(e.currentTarget).closest('.numberInput'),
            instance = $numberInput.data('numberInput');
            instance.updateTimeout && clearTimeout(instance.updateTimeout);
            instance.updateTimeout = setTimeout(function() {
                delete instance.updateTimeout;
                instance.value = parseFloat($numberInput.find('input').val());
                update($numberInput, instance);
            }, 1);
        } else if (e.keyCode == 38) {
            increment(e);
        } else if (e.keyCode == 40) {
            decrement(e);
        }
    }

    function change(e) {
        // Something changed, revalidate
        var $numberInput = $(e.currentTarget).closest('.numberInput'),
        instance = $numberInput.data('numberInput');
        instance.updateTimeout && clearTimeout(instance.updateTimeout);
        instance.updateTimeout = setTimeout(function() {
            delete instance.updateTimeout;
            instance.value = parseFloat($numberInput.find('input').val());
            update($numberInput, instance);
        }, 1);
    }

    function update($numberInput, instance, preventTrigger) {
        ensureBounds($numberInput, instance);
        $numberInput.find('input').val(String(instance.value));
        instance.$el.val(instance.value);
        if (!preventTrigger) {
            // trigger changes on the original element and the polyfill
            $numberInput.trigger("change");
            instance.$el.trigger("change");
        }
    }

    function ensureBounds($numberInput, instance) {
        updateBounds($numberInput);
        if (!isFinite(instance.value)) 
            instance.value = 0;
        instance.value = Math.max(Math.min(instance.value, instance.max), instance.min);
    }

    function updateBounds($numberInput) {
        var instance = $numberInput.data('numberInput'),
        $el = instance.$el;
        if (isFinite($el.attr("min"))) 
            instance.min = parseFloat($el.attr("min"));
        if (isFinite($el.attr("max"))) 
            instance.max = parseFloat($el.attr("max"));
    }


});

define("jquery.numberInput", function() {});

/* global jQuery:false, Modernizr:false */
(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define('internal/sitebuilder/common/jquery.loadingSpinner', ['jquery'], factory);
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function ($) {
    "use strict";
    /* jshint ignore:start */
    function loading() {
        var $this = $(this),
        instance = $this.data('loadingSpinner');
        if (instance.$spinner) {
            return;
        }
        instance.$spinner = $('<div/>').addClass('loadingSpinner').appendTo($this);
        instance.angle = 0;
        if ((typeof(Modernizr) !== 'undefined' && !Modernizr.cssanimations) || (typeof(Modernizr) == 'undefined' && $.browser.msie)) {
            instance.updateInterval = setInterval($.proxy(rotate_msie, $this), 81);
        }
    }

    function rotate_msie() {
        var instance = this.data('loadingSpinner');
        instance.angle = (instance.angle - 45) % 360;
        instance.$spinner.css({
            'msTransform': 'rotate(' + instance.angle + 'deg)'
        });
    }

    function loaded() {
        var $this = $(this),
        instance = $this.data('loadingSpinner');
        if (instance.updateInterval) {
            clearInterval(instance.updateInterval);
            delete instance.updateInterval;
        }
        if (instance.$spinner) {
            instance.$spinner.remove();
            delete instance.$spinner;
        }
    }

    function destroy() {
        var $this = $(this),
        instance = $this.data('loadingSpinner');
        $this.
        trigger('spinner.loaded').
        unbind('spinner.loading').
        unbind('spinner.loaded').
        unbind('spinner.destroy').
        data('loadingSpinner', null);
    }
    /* jshint ignore:end */

    $.fn.loadingSpinner = function loadingSpinner(opts) {
        var $this = $(this).eq(0),
        method;
        if (typeof opts == 'string') {
            method = opts;
            opts = arguments[1];
        }
        opts = $.extend({}, opts);
        $this.
        data('loadingSpinner', opts).
        bind('spinner.loading', loading).
        bind('spinner.loaded', loaded).
        bind('spinner.destroy', destroy);
        if (method === undefined || method === 'start') {
            $this.trigger('spinner.loading');
        }
        return this;
    };
}));
// TODO: Put this in a util library somewhere
Array.max = function( array ) {
    return Math.max.apply( Math, array );
};
Array.min = function( array ) {
    return Math.min.apply( Math, array );
};

(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define('jquery.cropper', ['jquery'], factory);
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function ($) {
    var cssToPx = function(el, attr) {
        var val = el.css(attr) || 0;

        if (typeof(val) === 'string') {
            if (val === 'auto') {
                val = 0;
            } else {
                var percentRelative = (attr === 'top' || attr === 'bottom') ? el.parent().height() : el.parent().width(); // what is the percent relative to?
                val = val.indexOf('%') > 0 ? Math.round(((parseFloat(val) / 100) * percentRelative)) : parseInt(val, 10);
            }
        }

        return val;
    };

    var methods = {
        init: function(opts) {
            var options = {
                reference: this,
                sides: ['top', 'right', 'bottom', 'left'],
                target: this.children().first(),
                centered: false,
                minWidth: 16,
                maxWidth: 3000,
                minHeight: 16,
                maxHeight: 3000,
                inBucket: false
            };

            if (typeof(opts) === 'object') 
                $.extend(options, opts);
            this.data('cropper', {
                options: options
            });

            var self = this,
            cropping = false,
            activeCropSide = null,
            hasChanged = false,
            startX, startY, startWidth, startHeight, startLeft, startTop,
            maxWidth, maxHeight, minWidth, minHeight,
            fullWidth, fullHeight, curWidth, curHeight,
            dragStart = function(e) {
                $(this).addClass("active");
                cropping = true;
                activeCropSide = $(this).data('side');
                $.extend(options, self.data('cropper').options);

                startX = e.clientX;
                startY = e.clientY;
                $(document.body).addClass('noselect');
                startLeft = cssToPx(options.target, 'left'); // crop left at the start of cropping
                startTop = cssToPx(options.target, 'top'); // crop top at the start of cropping
                startWidth = self.width(); // crop width at the start of cropping
                startHeight = self.height(); // crop height at the start of cropping
                fullWidth = options.target.width(); // width of the element being cropped
                fullHeight = options.target.height(); // height of the element being cropped
                curWidth = options.target.width(); // current width of the element being cropped
                curHeight = options.target.height(); // current height of the element being cropped
                minWidth = options.minWidth;
                minHeight = options.minHeight;

                /* max width and height must be the smallest height of linked images */
                var targetWidths = [options.maxWidth],
                targetHeights = [options.maxHeight];
                if (options.respectMaxDimensionsOnly) {
                    fullWidth = options.maxWidth;
                    fullHeight = options.maxHeight;
                } else {
                    for (var i = 0; i < options.target.length; i++) {
                        var img = $(options.target[i]),
                        _width = img.width(),
                        _height = img.height();

                        if (_width) 
                            targetWidths.push(_width);
                        if (_height) 
                            targetHeights.push(_height);
                    }
                }
                maxWidth = Array.min(targetWidths);
                maxHeight = Array.min(targetHeights);

                // set the position in pixels so that it doesn't move while we're cropping
                if (options.target.size() == 1) {
                    var pos = {};
                    if (startTop !== 0) 
                        pos.top = startTop;
                    if (startLeft !== 0) 
                        pos.left = startLeft;
                    options.target.css(pos);
                } else {
                    options.target.each(function(i, img) {
                        img = $(img);
                        var pos = {};
                        if (startTop !== 0) 
                            pos.top = cssToPx(img, 'top');
                        if (startLeft !== 0) 
                            pos.left = cssToPx(img, 'left');
                        img.css(pos);
                    });
                }


                if (options.onStart) 
                    options.onStart(activeCropSide);

                return false;
            },
            dragEnd = function(e) {
                if (cropping) {
                    $(document.body).removeClass('noselect');

                    if (options.onEnd) 
                        options.onEnd(activeCropSide, curWidth, curHeight);
                    if (hasChanged && options.onMoveEnd)
                        options.onMoveEnd(activeCropSide, curWidth, curHeight);

                    activeCropSide = null;
                    hasChanged = cropping = false;
                    self.cropHandles.removeClass("active");
                }
            },
            dragMove = function(e) {
                if (!cropping) 
                    return true;

                if (!hasChanged) {
                    if (typeof options.onStart == "function") 
                        options.onStart();
                    hasChanged = true;
                }

                var dx = (e.clientX - startX),
                dy = (e.clientY - startY);

                if (options.centered) 
                    dx = dx * 2;
                if (activeCropSide === 'right') {
                    var newWidth = startWidth + dx,
                    newLeft;

                    if (newWidth < minWidth) {
                        // too small
                        newWidth = minWidth;
                        dx = newWidth - startWidth;
                    } else if (newWidth > maxWidth) {
                        // too big
                        newWidth = maxWidth;
                        dx = newWidth - startWidth;

                        if (fullWidth < maxWidth) {
                            // trying to drag bigger than the actual image
                            newLeft = 0;
                        } else {
                            // trying to drag bigger than the container
                        }
                    } else {
                        // no bounds restrictions
                    }

                    if (newWidth > startLeft + fullWidth) {
                        // cannot pull from the right anymore. start bringing content from the left
                        newLeft = startLeft - ((startLeft + fullWidth) - newWidth);
                    }

                    if (typeof(newLeft) === 'number') 
                        options.target.css('left', newLeft + 'px');
                    options.target.width(options.target.width()); // convert to px (why?) (maybe this should happen externally, in an oncomplete)
                    self.width(newWidth);
                } else if (activeCropSide === 'left') {
                    dx = - dx;
                    var newWidth = startWidth + dx,
                    newLeft;

                    if (newWidth < minWidth) {
                        // too small
                        newWidth = minWidth;
                        dx = newWidth - startWidth;
                    } else if (newWidth > maxWidth) {
                        // too big
                        newWidth = maxWidth;
                        dx = newWidth - startWidth;

                        if (fullWidth < maxWidth) {
                            // trying to drag bigger than the actual image
                            newLeft = 0;
                        } else {
                            // trying to drag bigger than the container
                            newLeft = startLeft + dx;
                        }
                    } else {
                        // no bounds restrictions
                        // since we're pulling the left, adjust the css left
                        newLeft = startLeft + (newWidth - startWidth);
                    }

                    if (startLeft + dx > 0) {
                        // cannot pull from the left anymore. start bringing content from the right
                        newLeft = 0;
                    }

                    if (typeof(newLeft) === 'number') 
                        options.target.css('left', newLeft + 'px');
                    options.target.width(options.target.width()); // convert to px (why?) (maybe this should happen in an oncomplete)
                    self.width(newWidth);
                } else if (activeCropSide === 'bottom') {
                    var newHeight = startHeight + dy,
                    newTop;
                    newHeight = Math.max(newHeight, minHeight);

                    if (newHeight > maxHeight) {
                        // reached maxHeight set by whatever instantiated the cropper
                        newHeight = maxHeight;
                    } else if (newHeight > fullHeight) {
                        // trying to drag bigger than the actual image
                        newTop = true;
                        newHeight = fullHeight;
                    } else {
                        // no bounds restrictions
                        newTop = true;
                    }

                    // its possible we need to reduce one of the negative top positions of at least one of the images being cropped
                    if (newTop == true) {
                        options.target.each(function(i, img) {
                            img = $(img);
                            if (newHeight > cssToPx(img, 'top') + img.height()) {
                                newTop = cssToPx(img, 'top') - ((cssToPx(img, 'top') + img.height()) - newHeight);
                                img.css('top', newTop + 'px');
                            }
                        });
                    }
                    newTop = false;
                    self.height(newHeight);
                } else if (activeCropSide === 'top') {
                    dy = - dy;
                    var newHeight = startHeight + dy,
                    newTop;

                    newHeight = Math.max(newHeight, minHeight);
                    if (newHeight > fullHeight) {
                        // trying to drag bigger than the actual image
                        newTop = 0;
                        newHeight = fullHeight;
                    } else if (newHeight > startTop + fullHeight) {
                        // cannot pull from the bottom anymore. start bringing content from the top
                        newTop = startTop - ((startTop + fullHeight) - newHeight);
                    } else {
                        // no bounds restrictions
                        // since we're pulling the top, adjust the css top
                        newTop = startTop + (newHeight - startHeight);
                    }

                    if (startTop + dy > 0) {
                        // cannot pull from the top anymore. start bringing content from the bottom
                        newTop = 0;
                    }

                    if (typeof(newTop) === 'number') 
                        options.target.css('top', newTop + 'px');
                    self.height(newHeight);
                }

                curWidth = newWidth || startWidth;
                curHeight = newHeight || startHeight;
                if (options.onMove) 
                    options.onMove(activeCropSide, curWidth, curHeight);
            },
            // When the image changes, keep the width the same but uncrop to its full height
            imageChange = function(e) {
                var img = options.target,
                imgX = img.width(),
                imgY = img.height(),
                curX = self.width(),
                curY = self.height(),
                imgR = imgY / imgX,
                newY = imgR * curX;

                if (!options.inBucket) 
                    self.height(newY);
                if (options.onImageChange) 
                    options.onImageChange({
                        width: curX,
                        height: newY 
                    });
                e.stopPropagation();
            };

            // Create the handles
            $.each(['top', 'right', 'bottom', 'left'], function(i, side) {
                $('<div/>').addClass('bldr-crop-handle bldr-crop-handle-' + side).data('side', side).appendTo(options.reference).bind("mousedown.cropper", dragStart).toggle($.inArray(side, options.sides) >= 0);
            });
            self.cropHandles = options.reference.find(".bldr-crop-handle");

            $(document).bind('mousemove.cropper', dragMove).bind('mouseup.cropper', dragEnd);
            options.reference.bind('imageChange.cropper', imageChange);

            return this;
        },

        setOptions: function(opts) {
            var data = this.data('cropper');
            if (data && data.options) 
                $.extend(this.data('cropper').options, opts);
            return this;
        },

        destroy: function() {
            $(document).unbind('.cropper');
            $('.bldr-crop-handle').unbind(".cropper").remove();
            return this;
        }
    };

    $.fn.cropper = function(method) {
        if (methods[method]) {
            return methods[method].apply(this, Array.prototype.slice.call(arguments, 1));
        } else if (typeof(method) === 'object' || !method) {
            return methods.init.apply(this, arguments);
        } else {
            $.error('Method ' + method + ' does not exist on jQuery.cropper');
        }
        return this;
    };
}));
(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define('jquery.dragger', ['jquery'], factory);
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function ($) {
    var DRAGGABLE = '_draggable';
    var methods = {
        init: function(options) {

            document.ondragstart = function () {
                return false;
            }; // disable IE specialities

            var opts = $.extend({
                x: true,
                y: true,
                apply: true,
                escapeCondition: function() {
                    return false;
                }
            }, options),
            $dragEl = this.addClass(DRAGGABLE),
            $dragRef = opts.ref || $dragEl,
            dragging = false,
            $doc = $(document),
            props = {},
            dragStart = function(e) {
                var offset = $dragEl.position(),
                x0 = e.pageX - offset.left,
                y0 = e.pageY - offset.top;

                if (!dragging) {
                    dragging = true;
                    $dragEl.addClass('_dragging');
                    if (typeof opts.onStart === 'function') 
                        opts.onStart();
                }

                $doc.bind("mousemove.drag", function(e) {
                    var x = e.pageX - x0,
                    y = e.pageY - y0;

                    if (opts.x) 
                        props.left = x;
                    if (opts.y) 
                        props.top = y;

                    // Enables dragger to apply the css attributes
                    if (opts.apply && !opts.escapeCondition(props)) {
                        $dragEl.css(props);
                    }

                    // Trigger onMove
                    if (typeof opts.onMove === 'function') 
                        opts.onMove(props);
                });

                return false;
            },
            dragEnd = function() {
                try {
                    if (dragging) {
                        if (typeof opts.onEnd === 'function') 
                            opts.onEnd(props);
                    }
                } finally {
                    $doc.unbind("mousemove.drag");
                    dragging = false; //resets dragging flag
                    $dragEl.removeClass('_dragging');
                }
            },
            enable = function() {
                $dragRef.bind("mousedown.drag", dragStart);
                $doc.bind("mouseup.drag", dragEnd);
            },
            disable = function() {
                $dragRef.unbind("mousedown.drag");
                $doc.unbind("mouseup.drag");
                dragEnd();
            };


            enable();

            this.dragStart = dragStart;
            this.dragEnd = dragEnd;
            this.$doc = $doc;
            this.$dragEl = $dragEl;
            this.$dragRef = $dragRef;
            this.enable = enable;
            this.disable = disable;

            return $dragEl;
        },



        // For consistency purposes
        show: function() {
            return this.enable();
        },
        hide: function() {
            return this.disable();
        },

        destroy: function() {
            this.$dragRef.unbind(".drag").removeClass(DRAGGABLE);
            this.$doc.unbind(".drag");
        }
    };

    $.fn.dragger = function(method) {
        if (methods[method]) {
            return methods[method].apply(this, Array.prototype.slice.call(arguments, 1));
        } else if (typeof(method) === 'object' || !method) {
            return methods.init.apply(this, arguments);
        } else {
            $.error('Method ' + method + ' does not exist on jQuery.dragger');
        }
        return this;
    };
}));

(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define('jquery.panner', ['jquery'], factory);
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function ($) {
    var methods = {
        init: function(opts) {
            this.each(function() {
                var self = $(this);
                var options = {};
                if (typeof(opts) === 'object') 
                    $.extend(options, opts);
                self.data('panner', {
                    options: options
                });

                var panning = false,
                hasMoved = false,
                current = self,
                startX, startY,
                startTop, startLeft,
                width, height,
                bounds = {
                    top: 0,
                    left: 0,
                    right: 3000,
                    bottom: 3000
                },
                dragStart = function(e) {
                    current = $(this);
                    var pos = current.position(),
                    parent = current.parent();

                    $(document.body).addClass('panning').addClass('noselect');

                    panning = true;
                    $.extend(options, current.data('panner').options);

                    startTop = pos.top;
                    startLeft = pos.left;

                    width = current.width();
                    height = current.height();

                    bounds.right = parent.width();

                    // In certain styling situations, .height doesn't return what we want
                    bounds.bottom = parent.height() > 1 ? parent.height() : parent.outerHeight();

                    startX = e.clientX;
                    startY = e.clientY;

                    if (typeof(options.onStart) === 'function') 
                        options.onStart(startX, startY);

                    return false;
                },
                dragEnd = function(e) {
                    if (panning) {
                        $(document.body).removeClass('panning').removeClass('noselect').css('cursor', 'auto');
                        current.removeClass('panning');

                        if (typeof(options.onEnd) === 'function') 
                            options.onEnd();

                        hasMoved = panning = false;
                    }
                },
                dragMove = function(e) {
                    if (!panning) 
                        return true;

                    var dx = (e.clientX - startX),
                    dy = (e.clientY - startY),
                    newLeft = startLeft + dx,
                    newTop = startTop + dy;

                    if (!hasMoved) {
                        if (typeof(options.onStart) === 'function') 
                            options.onStart();
                        hasMoved = true;
                    }

                    if (newTop + height < bounds.bottom) 
                        newTop = 0 - (height - bounds.bottom);
                    if (newTop > 0) 
                        newTop = 0;
                    if (newLeft + width < bounds.right) 
                        newLeft = 0 - (width - bounds.right);
                    if (newLeft > 0) 
                        newLeft = 0;

                    current.css({
                        top: newTop,
                        left: newLeft 
                    });

                    if (typeof(options.onMove) === 'function') 
                        options.onMove(newTop, newLeft, dy, dx);
                },
                imageChange = function(e) {
                    current.css({
                        top: 0,
                        left: 0
                    });
                    e.stopPropagation();
                };

                self.addClass('pannable').bind('mousedown.panner', dragStart);

                $(document).bind('mousemove.panner', dragMove).bind('mouseup.panner', dragEnd);
                options.reference.bind('imageChange.panner', imageChange);
            });

            return this;
        },

        setOptions: function(opts) {
            var data = this.data('panner');
            if (data && data.options) 
                $.extend(this.data('panner').options, opts);
            return this;
        },

        destroy: function() {
            var dat = this.data("panner");
            if (dat && dat.options && dat.options.reference) 
                dat.options.reference.unbind(".panner");
            this.removeClass('pannable').unbind('.panner');
            $(document).unbind(".panner");
            return this;
        }
    };

    $.fn.panner = function(method) {
        if (methods[method]) {
            return methods[method].apply(this, Array.prototype.slice.call(arguments, 1));
        } else if (typeof(method) === 'object' || !method) {
            return methods.init.apply(this, arguments);
        } else {
            $.error('Method ' + method + ' does not exist on jQuery.panner');
        }
        return this;
    };
}));
(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define('jquery.resizer', ['jquery'], factory);
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function ($) {
    var methods = {
        init: function(opts) {
            var options = {
                reference: this,
                corners: ['left', 'right'],
                target: this.children().first(),
                centered: false,
                maxWidth: 3000,
                maxHeight: 3000,
                lockRatio: true,
                minWidth: 16,
                minHeight: 16
            };
            if (typeof(opts) === 'object') 
                $.extend(options, opts);
            this.data('resizer', {
                options: options
            });

            var self = this,
            resizing = false,
            activeCorner = null,
            startX, startY, startWidth, startHeight,
            aspectRatio,
            dragStart = function(e) {
                $(this).addClass("active");
                resizing = true;
                activeCorner = $(this).data('corner');
                $.extend(options, self.data('resizer').options);

                if (options.onStart) 
                    options.onStart(activeCorner);

                startX = e.clientX;
                startY = e.clientY;
                $(document.body).addClass('noselect');
                startWidth = self.width(); // crop width at the start of cropping
                startHeight = self.height(); // crop height at the start of cropping
                aspectRatio = (startWidth / startHeight);

                return false;
            },
            dragEnd = function(e) {
                if (resizing) {
                    $(document.body).removeClass('noselect');

                    if (options.onEnd) 
                        options.onEnd(activeCorner);

                    activeCorner = null;
                    resizing = false;
                    self.resizeHandles.removeClass("active");
                }
            },
            dragMove = function(e) {
                if (!resizing) 
                    return;

                var dx = (e.clientX - startX);
                var dy = (e.clientY - startY);

                if (options.centered) 
                    dx = dx * 2;
                if (activeCorner === 'left') 
                    dx = - dx;
                var newWidth = startWidth + dx, newHeight;

                // Ensure its not too big or small
                if (options.lockRatio) {
                    newWidth = parseInt(Math.min(newWidth, options.maxWidth, aspectRatio * options.maxHeight, options.maxHeight));
                    newWidth = parseInt(Math.max(newWidth, options.minWidth, aspectRatio * options.minHeight));
                    newHeight = parseInt(newWidth / aspectRatio);
                } else {
                    newWidth = parseInt(Math.min(newWidth, options.maxWidth));
                    newHeight = parseInt(startHeight + dy);
                }
                if (newWidth < 1) 
                    newWidth = 1;
                if (newHeight < 1) 
                    newHeight = 1;
                self.width(newWidth).height(newHeight);

                if (options.onMove) 
                    options.onMove(activeCorner, newWidth, newHeight);
            };

            // Create the handles
            $.each(['left', 'right'], function(i, corner) {
                if ($.inArray(corner, options.corners) >= 0)
                    $('<div/>').addClass('bldr-resize-handle bldr-resize-handle-' + corner).data('corner', corner).appendTo(options.reference).bind("mousedown.resizer", dragStart);
            });
            self.resizeHandles = options.reference.find('.bldr-resize-handle');

            $(document).bind('mousemove.resizer', dragMove).bind('mouseup.resizer', dragEnd);

            return this;
        },

        setOptions: function(opts) {
            var data = this.data('resizer');
            if (data && data.options) 
                $.extend(this.data('resizer').options, opts);
            return this;
        },

        hide: function() {
            if (this.resizeHandles) {
                this.resizeHandles.hide();
                return true;
            } else {
                return false;
            }
        },

        show: function() {
            if (this.resizeHandles) {
                this.resizeHandles.show();
                return true;
            } else {
                return false;
            }
        },

        destroy: function() {
            $(document).unbind('.resizer');
            $('.bldr-resize-handle').unbind(".resizer").remove();
            return this;
        }
    };

    $.fn.resizer = function(method) {
        if (methods[method]) {
            return methods[method].apply(this, Array.prototype.slice.call(arguments, 1));
        } else if (typeof(method) === 'object' || !method) {
            return methods.init.apply(this, arguments);
        } else {
            $.error('Method ' + method + ' does not exist on jQuery.resizer');
            return null;
        }
        return this;
    };
}));
(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define('jquery.zoomer', ['jquery'], factory);
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function ($) {
    var methods = {
        init: function(opts) {
            var options = {
                reference: this,
                min: 0,
                max: 3000,
                step: 1
            };
            if (typeof(opts) === 'object') 
                $.extend(options, opts);

            var self = this,
            zoomRange;

            zoomRange = $('<input/>').attr({
                type: 'range',
                min: options.min,
                max: options.max,
                step: options.step
            }).val(this.width()).appendTo($('<div/>').addClass('bldr-image-zoomer').appendTo(options.reference)).customrange({
                dragStart: options.dragStart,
                dragEnd: options.dragEnd
            });

            changing = false;
            zoomRange.change(function(e) {
                if (!changing) {
                    changing = true;
                    setTimeout(function() {
                        changing = false;
                    }, 1000);
                    if (typeof(options.onStart) === 'function') 
                        options.onStart();
                }
                if (typeof(options.onChange) === 'function') 
                    options.onChange(zoomRange.val());
            });

            options.reference.bind('imageChange.zoomer', function() {
                zoomRange.val(1).change();
            });

            this.data('zoomer', {
                options: options,
                el: zoomRange.parent()
            });

            return this;
        },

        destroy: function() {
            var zoomer = this.data("zoomer");
            if (zoomer) 
                zoomer.el.remove();

            // TODO: Destroy the customrange (which should destroy any global events
            return this;
        }
    };

    $.fn.zoomer = function(method) {
        if (methods[method]) {
            return methods[method].apply(this, Array.prototype.slice.call(arguments, 1));
        } else if (typeof(method) === 'object' || !method) {
            return methods.init.apply(this, arguments);
        } else {
            $.error('Method ' + method + ' does not exist on jQuery.zoomer');
        }
        return this;
    };
}));

/* jshint ignore:start */
/*
 Rangy, a cross-browser JavaScript range and selection library
 http://code.google.com/p/rangy/

 Copyright 2012, Tim Down
 Licensed under the MIT license.
 Version: 1.2.3
 Build date: 26 February 2012
*/
window.rangy = function() {
    function l(p, u) {
        var w = typeof p[u];
        return w == "function"||!!(w == "object" && p[u]) || w == "unknown"
    }
    function K(p, u) {
        return !!(typeof p[u] == "object" && p[u])
    }
    function H(p, u) {
        return typeof p[u] != "undefined"
    }
    function I(p) {
        return function(u, w) {
            for (var B = w.length; B--;)
                if (!p(u, w[B]))
                    return false;
            return true
        }
    }
    function z(p) {
        return p && A(p, x) && v(p, t)
    }
    function C(p) {
        window.alert("Rangy not supported in your browser. Reason: " + p);
        c.initialized = true;
        c.supported = false
    }
    function N() {
        if (!c.initialized) {
            var p,
            u = false, w = false;
            if (l(document, "createRange")) {
                p = document.createRange();
                if (A(p, n) && v(p, i))
                    u = true;
                p.detach()
            }
            if ((p = K(document, "body") ? document.body : document.getElementsByTagName("body")[0]) && l(p, "createTextRange")) {
                p = p.createTextRange();
                if (z(p))
                    w = true
            }
            !u&&!w && C("Neither Range nor TextRange are implemented");
            c.initialized = true;
            c.features = {
                implementsDomRange: u,
                implementsTextRange: w
            };
            u = k.concat(f);
            w = 0;
            for (p = u.length; w < p; ++w)
                try {
                    u[w](c)
                } catch (B) {
                K(window, "console") && l(window.console, "log") && window.console.log("Init listener threw an exception. Continuing.",
                B)
            }
        }
    }
    function O(p) {
        this.name = p;
        this.supported = this.initialized = false
    }
    var i = ["startContainer", "startOffset", "endContainer", "endOffset", "collapsed", "commonAncestorContainer", "START_TO_START", "START_TO_END", "END_TO_START", "END_TO_END"], n = ["setStart", "setStartBefore", "setStartAfter", "setEnd", "setEndBefore", "setEndAfter", "collapse", "selectNode", "selectNodeContents", "compareBoundaryPoints", "deleteContents", "extractContents", "cloneContents", "insertNode", "surroundContents", "cloneRange", "toString", "detach"],
    t = ["boundingHeight", "boundingLeft", "boundingTop", "boundingWidth", "htmlText", "text"], x = ["collapse", "compareEndPoints", "duplicate", "getBookmark", "moveToBookmark", "moveToElementText", "parentElement", "pasteHTML", "select", "setEndPoint", "getBoundingClientRect"], A = I(l), q = I(K), v = I(H), c = {
        version: "1.2.3",
        initialized: false,
        supported: true,
        util: {
            isHostMethod: l,
            isHostObject: K,
            isHostProperty: H,
            areHostMethods: A,
            areHostObjects: q,
            areHostProperties: v,
            isTextRange: z
        },
        features: {},
        modules: {},
        config: {
            alertOnWarn: false,
            preferTextRange: false
        }
    };
    c.fail = C;
    c.warn = function(p) {
        p = "Rangy warning: " + p;
        if (c.config.alertOnWarn)
            window.alert(p);
        else 
            typeof window.console != "undefined" && typeof window.console.log != "undefined" && window.console.log(p)
    };
    if ({}.hasOwnProperty)
        c.util.extend = function(p, u) {
            for (var w in u)
                if (u.hasOwnProperty(w))
                    p[w] = u[w]
                };
            else 
                C("hasOwnProperty not supported");
    var f = [], k = [];
    c.init = N;
    c.addInitListener = function(p) {
        c.initialized ? p(c) : f.push(p)
    };
    var r = [];
    c.addCreateMissingNativeApiListener = function(p) {
        r.push(p)
    };
    c.createMissingNativeApi =
    function(p) {
        p = p || window;
        N();
        for (var u = 0, w = r.length; u < w; ++u)
            r[u](p)
    };
    O.prototype.fail = function(p) {
        this.initialized = true;
        this.supported = false;
        throw Error("Module '" + this.name + "' failed to load: " + p);
    };
    O.prototype.warn = function(p) {
        c.warn("Module " + this.name + ": " + p)
    };
    O.prototype.createError = function(p) {
        return Error("Error in Rangy " + this.name + " module: " + p)
    };
    c.createModule = function(p, u) {
        var w = new O(p);
        c.modules[p] = w;
        k.push(function(B) {
            u(B, w);
            w.initialized = true;
            w.supported = true
        })
    };
    c.requireModules = function(p) {
        for (var u =
        0, w = p.length, B, V; u < w; ++u) {
            V = p[u];
            B = c.modules[V];
            if (!B ||!(B instanceof O))
                throw Error("Module '" + V + "' not found");
            if (!B.supported)
                throw Error("Module '" + V + "' not supported");
        }
    };
    var L = false;
    q = function() {
        if (!L) {
            L = true;
            c.initialized || N()
        }
    };
    if (typeof window == "undefined")
        C("No window found");
    else if (typeof document == "undefined")
        C("No document found");
    else {
        l(document, "addEventListener") && document.addEventListener("DOMContentLoaded", q, false);
        if (l(window, "addEventListener"))
            window.addEventListener("load",
            q, false);
        else 
            l(window, "attachEvent") ? window.attachEvent("onload", q) : C("Window does not have required addEventListener or attachEvent method");
        return c
    }
}();
rangy.createModule("DomUtil", function(l, K) {
    function H(c) {
        for (var f = 0; c = c.previousSibling;)
            f++;
        return f
    }
    function I(c, f) {
        var k = [], r;
        for (r = c; r; r = r.parentNode)
            k.push(r);
        for (r = f; r; r = r.parentNode)
            if (v(k, r))
                return r;
        return null
    }
    function z(c, f, k) {
        for (k = k ? c : c.parentNode; k;) {
            c = k.parentNode;
            if (c === f)
                return k;
            k = c
        }
        return null
    }
    function C(c) {
        c = c.nodeType;
        return c == 3 || c == 4 || c == 8
    }
    function N(c, f) {
        var k = f.nextSibling, r = f.parentNode;
        k ? r.insertBefore(c, k) : r.appendChild(c);
        return c
    }
    function O(c) {
        if (c.nodeType == 9)
            return c;
        else if (typeof c.ownerDocument != "undefined")
            return c.ownerDocument;
        else if (typeof c.document != "undefined")
            return c.document;
        else if (c.parentNode)
            return O(c.parentNode);
        else 
            throw Error("getDocument: no document found for node");
    }
    function i(c) {
        if (!c)
            return "[No node]";
        return C(c) ? '"' + c.data + '"' : c.nodeType == 1 ? "<" + c.nodeName + (c.id ? ' id="' + c.id + '"' : "") + ">[" + c.childNodes.length + "]" : c.nodeName
    }
    function n(c) {
        this._next = this.root = c
    }
    function t(c, f) {
        this.node = c;
        this.offset = f
    }
    function x(c) {
        this.code = this[c];
        this.codeName = c;
        this.message = "DOMException: " + this.codeName
    }
    var A = l.util;
    A.areHostMethods(document, ["createDocumentFragment", "createElement", "createTextNode"]) || K.fail("document missing a Node creation method");
    A.isHostMethod(document, "getElementsByTagName") || K.fail("document missing getElementsByTagName method");
    var q = document.createElement("div");
    A.areHostMethods(q, ["insertBefore", "appendChild", "cloneNode"]) || K.fail("Incomplete Element implementation");
    A.isHostProperty(q, "innerHTML") || K.fail("Element is missing innerHTML property");
    q = document.createTextNode("test");
    A.areHostMethods(q, ["splitText", "deleteData", "insertData", "appendData", "cloneNode"]) || K.fail("Incomplete Text Node implementation");
    var v = function(c, f) {
        for (var k = c.length; k--;)
            if (c[k] === f)
                return true;
        return false
    };
    n.prototype = {
        _current: null,
        hasNext: function() {
            return !!this._next
        },
        next: function() {
            var c = this._current = this._next, f;
            if (this._current)
                if (f = c.firstChild)
                    this._next = f;
                else {
                    for (f = null; c !== this.root&&!(f = c.nextSibling);)
                        c = c.parentNode;
                        this._next = f
                }
            return this._current
        },
        detach: function() {
            this._current = this._next = this.root = null
        }
    };
    t.prototype = {
        equals: function(c) {
            return this.node === c.node & this.offset == c.offset
        },
        inspect: function() {
            return "[DomPosition(" + i(this.node) + ":" + this.offset + ")]"
        }
    };
    x.prototype = {
        INDEX_SIZE_ERR: 1,
        HIERARCHY_REQUEST_ERR: 3,
        WRONG_DOCUMENT_ERR: 4,
        NO_MODIFICATION_ALLOWED_ERR: 7,
        NOT_FOUND_ERR: 8,
        NOT_SUPPORTED_ERR: 9,
        INVALID_STATE_ERR: 11
    };
    x.prototype.toString = function() {
        return this.message
    };
    l.dom = {
        arrayContains: v,
        isHtmlNamespace: function(c) {
            var f;
            return typeof c.namespaceURI ==
            "undefined" || (f = c.namespaceURI) === null || f == "http://www.w3.org/1999/xhtml"
        },
        parentElement: function(c) {
            c = c.parentNode;
            return c.nodeType == 1 ? c : null
        },
        getNodeIndex: H,
        getNodeLength: function(c) {
            var f;
            return C(c) ? c.length : (f = c.childNodes) ? f.length : 0
        },
        getCommonAncestor: I,
        isAncestorOf: function(c, f, k) {
            for (f = k ? f : f.parentNode; f;)
                if (f === c)
                    return true;
                else 
                    f = f.parentNode;
            return false
        },
        getClosestAncestorIn: z,
        isCharacterDataNode: C,
        insertAfter: N,
        splitDataNode: function(c, f) {
            var k = c.cloneNode(false);
            k.deleteData(0, f);
            c.deleteData(f, c.length - f);
            N(k, c);
            return k
        },
        getDocument: O,
        getWindow: function(c) {
            c = O(c);
            if (typeof c.defaultView != "undefined")
                return c.defaultView;
            else if (typeof c.parentWindow != "undefined")
                return c.parentWindow;
            else 
                throw Error("Cannot get a window object for node");
        },
        getIframeWindow: function(c) {
            if (typeof c.contentWindow != "undefined")
                return c.contentWindow;
            else if (typeof c.contentDocument != "undefined")
                return c.contentDocument.defaultView;
            else 
                throw Error("getIframeWindow: No Window object found for iframe element");
        },
        getIframeDocument: function(c) {
            if (typeof c.contentDocument != "undefined")
                return c.contentDocument;
            else if (typeof c.contentWindow != "undefined")
                return c.contentWindow.document;
            else 
                throw Error("getIframeWindow: No Document object found for iframe element");
        },
        getBody: function(c) {
            return A.isHostObject(c, "body") ? c.body : c.getElementsByTagName("body")[0]
        },
        getRootContainer: function(c) {
            for (var f; f = c.parentNode;)
                c = f;
            return c
        },
        comparePoints: function(c, f, k, r) {
            var L;
            if (c == k)
                return f === r ? 0 : f < r?-1 : 1;
            else if (L = z(k,
            c, true))
                return f <= H(L)?-1 : 1;
            else if (L = z(c, k, true))
                return H(L) < r?-1 : 1;
            else {
                f = I(c, k);
                c = c === f ? f : z(c, f, true);
                k = k === f ? f : z(k, f, true);
                if (c === k)
                    throw Error("comparePoints got to case 4 and childA and childB are the same!");
                else {
                    for (f = f.firstChild; f;) {
                        if (f === c)
                            return - 1;
                        else if (f === k)
                            return 1;
                        f = f.nextSibling
                    }
                    throw Error("Should not be here!");
                }
            }
        },
        inspectNode: i,
        fragmentFromNodeChildren: function(c) {
            for (var f = O(c).createDocumentFragment(), k; k = c.firstChild;)
                f.appendChild(k);
            return f
        },
        createIterator: function(c) {
            return new n(c)
        },
        DomPosition: t
    };
    l.DOMException = x
});
rangy.createModule("DomRange", function(l) {
    function K(a, e) {
        return a.nodeType != 3 && (g.isAncestorOf(a, e.startContainer, true) || g.isAncestorOf(a, e.endContainer, true))
    }
    function H(a) {
        return g.getDocument(a.startContainer)
    }
    function I(a, e, j) {
        if (e = a._listeners[e])
            for (var o = 0, E = e.length; o < E; ++o)
                e[o].call(a, {
                    target: a,
                    args: j
                })
    }
    function z(a) {
        return new Z(a.parentNode, g.getNodeIndex(a))
    }
    function C(a) {
        return new Z(a.parentNode, g.getNodeIndex(a) + 1)
    }
    function N(a, e, j) {
        var o = a.nodeType == 11 ? a.firstChild: a;
        if (g.isCharacterDataNode(e))
            j ==
            e.length ? g.insertAfter(a, e) : e.parentNode.insertBefore(a, j == 0 ? e : g.splitDataNode(e, j));
        else 
            j >= e.childNodes.length ? e.appendChild(a) : e.insertBefore(a, e.childNodes[j]);
        return o
    }
    function O(a) {
        for (var e, j, o = H(a.range).createDocumentFragment(); j = a.next();) {
            e = a.isPartiallySelectedSubtree();
            j = j.cloneNode(!e);
            if (e) {
                e = a.getSubtreeIterator();
                j.appendChild(O(e));
                e.detach(true)
            }
            if (j.nodeType == 10)
                throw new S("HIERARCHY_REQUEST_ERR");
            o.appendChild(j)
        }
        return o
    }
    function i(a, e, j) {
        var o, E;
        for (j = j || {
            stop: false
        }; o = a.next();)
            if (a.isPartiallySelectedSubtree())
                if (e(o) ===
                false) {
                    j.stop = true;
                    return 
                } else {
                    o = a.getSubtreeIterator();
                    i(o, e, j);
                    o.detach(true);
                    if (j.stop)
                        return 
                } else 
                    for (o = g.createIterator(o); E = o.next();)
                        if (e(E) === false) {
                            j.stop = true;
                            return 
                        }
    }
    function n(a) {
        for (var e; a.next();)
            if (a.isPartiallySelectedSubtree()) {
                e = a.getSubtreeIterator();
                n(e);
                e.detach(true)
            } else 
                a.remove()
    }
    function t(a) {
        for (var e, j = H(a.range).createDocumentFragment(), o; e = a.next();) {
            if (a.isPartiallySelectedSubtree()) {
                e = e.cloneNode(false);
                o = a.getSubtreeIterator();
                e.appendChild(t(o));
                o.detach(true)
            } else 
                a.remove();
            if (e.nodeType == 10)
                throw new S("HIERARCHY_REQUEST_ERR");
            j.appendChild(e)
        }
        return j
    }
    function x(a, e, j) {
        var o=!!(e && e.length), E, T=!!j;
        if (o)
            E = RegExp("^(" + e.join("|") + ")$");
        var m = [];
        i(new q(a, false), function(s) {
            if ((!o || E.test(s.nodeType)) && (!T || j(s)))
                m.push(s)
        });
        return m
    }
    function A(a) {
        return "[" + (typeof a.getName == "undefined" ? "Range" : a.getName()) + "(" + g.inspectNode(a.startContainer) + ":" + a.startOffset + ", " + g.inspectNode(a.endContainer) + ":" + a.endOffset + ")]"
    }
    function q(a, e) {
        this.range = a;
        this.clonePartiallySelectedTextNodes =
        e;
        if (!a.collapsed) {
            this.sc = a.startContainer;
            this.so = a.startOffset;
            this.ec = a.endContainer;
            this.eo = a.endOffset;
            var j = a.commonAncestorContainer;
            if (this.sc === this.ec && g.isCharacterDataNode(this.sc)) {
                this.isSingleCharacterDataNode = true;
                this._first = this._last = this._next = this.sc
            } else {
                this._first = this._next = this.sc === j&&!g.isCharacterDataNode(this.sc) ? this.sc.childNodes[this.so] : g.getClosestAncestorIn(this.sc, j, true);
                this._last = this.ec === j&&!g.isCharacterDataNode(this.ec) ? this.ec.childNodes[this.eo - 1] : g.getClosestAncestorIn(this.ec,
                j, true)
            }
        }
    }
    function v(a) {
        this.code = this[a];
        this.codeName = a;
        this.message = "RangeException: " + this.codeName
    }
    function c(a, e, j) {
        this.nodes = x(a, e, j);
        this._next = this.nodes[0];
        this._position = 0
    }
    function f(a) {
        return function(e, j) {
            for (var o, E = j ? e : e.parentNode; E;) {
                o = E.nodeType;
                if (g.arrayContains(a, o))
                    return E;
                E = E.parentNode
            }
            return null
        }
    }
    function k(a, e) {
        if (G(a, e))
            throw new v("INVALID_NODE_TYPE_ERR");
    }
    function r(a) {
        if (!a.startContainer)
            throw new S("INVALID_STATE_ERR");
    }
    function L(a, e) {
        if (!g.arrayContains(e, a.nodeType))
            throw new v("INVALID_NODE_TYPE_ERR");
    }
    function p(a, e) {
        if (e < 0 || e > (g.isCharacterDataNode(a) ? a.length : a.childNodes.length))
            throw new S("INDEX_SIZE_ERR");
    }
    function u(a, e) {
        if (h(a, true) !== h(e, true))
            throw new S("WRONG_DOCUMENT_ERR");
    }
    function w(a) {
        if (D(a, true))
            throw new S("NO_MODIFICATION_ALLOWED_ERR");
    }
    function B(a, e) {
        if (!a)
            throw new S(e);
    }
    function V(a) {
        return !!a.startContainer&&!!a.endContainer&&!(!g.arrayContains(ba, a.startContainer.nodeType)&&!h(a.startContainer, true))&&!(!g.arrayContains(ba, a.endContainer.nodeType)&&!h(a.endContainer,
        true)) && a.startOffset <= (g.isCharacterDataNode(a.startContainer) ? a.startContainer.length : a.startContainer.childNodes.length) && a.endOffset <= (g.isCharacterDataNode(a.endContainer) ? a.endContainer.length : a.endContainer.childNodes.length)
    }
    function J(a) {
        r(a);
        if (!V(a))
            throw Error("Range error: Range is no longer valid after DOM mutation (" + a.inspect() + ")");
    }
    function ca() {}
    function Y(a) {
        a.START_TO_START = ia;
        a.START_TO_END = la;
        a.END_TO_END = ra;
        a.END_TO_START = ma;
        a.NODE_BEFORE = na;
        a.NODE_AFTER = oa;
        a.NODE_BEFORE_AND_AFTER =
        pa;
        a.NODE_INSIDE = ja
    }
    function W(a) {
        Y(a);
        Y(a.prototype)
    }
    function da(a, e) {
        return function() {
            J(this);
            var j = this.startContainer, o = this.startOffset, E = this.commonAncestorContainer, T = new q(this, true);
            if (j !== E) {
                j = g.getClosestAncestorIn(j, E, true);
                o = C(j);
                j = o.node;
                o = o.offset
            }
            i(T, w);
            T.reset();
            E = a(T);
            T.detach();
            e(this, j, o, j, o);
            return E
        }
    }
    function fa(a, e, j) {
        function o(m, s) {
            return function(y) {
                r(this);
                L(y, $);
                L(d(y), ba);
                y = (m ? z : C)(y);
                (s ? E : T)(this, y.node, y.offset)
            }
        }
        function E(m, s, y) {
            var F = m.endContainer, Q = m.endOffset;
            if (s !== m.startContainer || y !== m.startOffset) {
                if (d(s) != d(F) || g.comparePoints(s, y, F, Q) == 1) {
                    F = s;
                    Q = y
                }
                e(m, s, y, F, Q)
            }
        }
        function T(m, s, y) {
            var F = m.startContainer, Q = m.startOffset;
            if (s !== m.endContainer || y !== m.endOffset) {
                if (d(s) != d(F) || g.comparePoints(s, y, F, Q)==-1) {
                    F = s;
                    Q = y
                }
                e(m, F, Q, s, y)
            }
        }
        a.prototype = new ca;
        l.util.extend(a.prototype, {
            setStart: function(m, s) {
                r(this);
                k(m, true);
                p(m, s);
                E(this, m, s)
            },
            setEnd: function(m, s) {
                r(this);
                k(m, true);
                p(m, s);
                T(this, m, s)
            },
            setStartBefore: o(true, true),
            setStartAfter: o(false, true),
            setEndBefore: o(true,
            false),
            setEndAfter: o(false, false),
            collapse: function(m) {
                J(this);
                m ? e(this, this.startContainer, this.startOffset, this.startContainer, this.startOffset) : e(this, this.endContainer, this.endOffset, this.endContainer, this.endOffset)
            },
            selectNodeContents: function(m) {
                r(this);
                k(m, true);
                e(this, m, 0, m, g.getNodeLength(m))
            },
            selectNode: function(m) {
                r(this);
                k(m, false);
                L(m, $);
                var s = z(m);
                m = C(m);
                e(this, s.node, s.offset, m.node, m.offset)
            },
            extractContents: da(t, e),
            deleteContents: da(n, e),
            canSurroundContents: function() {
                J(this);
                w(this.startContainer);
                w(this.endContainer);
                var m = new q(this, true), s = m._first && K(m._first, this) || m._last && K(m._last, this);
                m.detach();
                return !s
            },
            detach: function() {
                j(this)
            },
            splitBoundaries: function() {
                J(this);
                var m = this.startContainer, s = this.startOffset, y = this.endContainer, F = this.endOffset, Q = m === y;
                g.isCharacterDataNode(y) && F > 0 && F < y.length && g.splitDataNode(y, F);
                if (g.isCharacterDataNode(m) && s > 0 && s < m.length) {
                    m = g.splitDataNode(m, s);
                    if (Q) {
                        F -= s;
                        y = m
                    } else 
                        y == m.parentNode && F >= g.getNodeIndex(m) && F++;
                    s = 0
                }
                e(this, m, s, y, F)
            },
            normalizeBoundaries: function() {
                J(this);
                var m = this.startContainer, s = this.startOffset, y = this.endContainer, F = this.endOffset, Q = function(U) {
                    var R = U.nextSibling;
                    if (R && R.nodeType == U.nodeType) {
                        y = U;
                        F = U.length;
                        U.appendData(R.data);
                        R.parentNode.removeChild(R)
                    }
                }, qa = function(U) {
                    var R = U.previousSibling;
                    if (R && R.nodeType == U.nodeType) {
                        m = U;
                        var sa = U.length;
                        s = R.length;
                        U.insertData(0, R.data);
                        R.parentNode.removeChild(R);
                        if (m == y) {
                            F += s;
                            y = m
                        } else if (y == U.parentNode) {
                            R = g.getNodeIndex(U);
                            if (F == R) {
                                y = U;
                                F = sa
                            } else 
                                F > R && F--
                        }
                    }
                }, ga = true;
                if (g.isCharacterDataNode(y))
                    y.length ==
                    F && Q(y);
                else {
                    if (F > 0)(ga = y.childNodes[F - 1]) && g.isCharacterDataNode(ga) && Q(ga);
                    ga=!this.collapsed
                }
                if (ga)
                    if (g.isCharacterDataNode(m))
                        s == 0 && qa(m);
                    else {
                        if (s < m.childNodes.length)(Q = m.childNodes[s]) && g.isCharacterDataNode(Q) && qa(Q)
                        } else {
                            m = y;
                            s = F
                        }
                        e(this, m, s, y, F)
                }, collapseToPoint: function(m, s) {
                r(this);
                k(m, true);
                p(m, s);
                if (m !== this.startContainer || s !== this.startOffset || m !== this.endContainer || s !== this.endOffset)
                    e(this, m, s, m, s)
            }
        });
        W(a)
    }
    function ea(a) {
        a.collapsed = a.startContainer === a.endContainer && a.startOffset ===
        a.endOffset;
        a.commonAncestorContainer = a.collapsed ? a.startContainer : g.getCommonAncestor(a.startContainer, a.endContainer)
    }
    function ha(a, e, j, o, E) {
        var T = a.startContainer !== e || a.startOffset !== j, m = a.endContainer !== o || a.endOffset !== E;
        a.startContainer = e;
        a.startOffset = j;
        a.endContainer = o;
        a.endOffset = E;
        ea(a);
        I(a, "boundarychange", {
            startMoved: T,
            endMoved: m
        })
    }
    function M(a) {
        this.startContainer = a;
        this.startOffset = 0;
        this.endContainer = a;
        this.endOffset = 0;
        this._listeners = {
            boundarychange: [],
            detach: []
        };
        ea(this)
    }
    l.requireModules(["DomUtil"]);
    var g = l.dom, Z = g.DomPosition, S = l.DOMException;
    q.prototype = {
        _current: null,
        _next: null,
        _first: null,
        _last: null,
        isSingleCharacterDataNode: false,
        reset: function() {
            this._current = null;
            this._next = this._first
        },
        hasNext: function() {
            return !!this._next
        },
        next: function() {
            var a = this._current = this._next;
            if (a) {
                this._next = a !== this._last ? a.nextSibling : null;
                if (g.isCharacterDataNode(a) && this.clonePartiallySelectedTextNodes) {
                    if (a === this.ec)(a = a.cloneNode(true)).deleteData(this.eo, a.length - this.eo);
                    if (this._current === this.sc)(a =
                    a.cloneNode(true)).deleteData(0, this.so)
                    }
            }
            return a
        }, remove: function() {
            var a = this._current, e, j;
            if (g.isCharacterDataNode(a) && (a === this.sc || a === this.ec)) {
                e = a === this.sc ? this.so : 0;
                j = a === this.ec ? this.eo : a.length;
                e != j && a.deleteData(e, j - e)
            } else 
                a.parentNode && a.parentNode.removeChild(a)
        }, isPartiallySelectedSubtree: function() {
            return K(this._current, this.range)
        }, getSubtreeIterator: function() {
            var a;
            if (this.isSingleCharacterDataNode) {
                a = this.range.cloneRange();
                a.collapse()
            } else {
                a = new M(H(this.range));
                var e = this._current,
                j = e, o = 0, E = e, T = g.getNodeLength(e);
                if (g.isAncestorOf(e, this.sc, true)) {
                    j = this.sc;
                    o = this.so
                }
                if (g.isAncestorOf(e, this.ec, true)) {
                    E = this.ec;
                    T = this.eo
                }
                ha(a, j, o, E, T)
            }
            return new q(a, this.clonePartiallySelectedTextNodes)
        }, detach: function(a) {
            a && this.range.detach();
            this.range = this._current = this._next = this._first = this._last = this.sc = this.so = this.ec = this.eo = null
        }
    };
    v.prototype = {
        BAD_BOUNDARYPOINTS_ERR: 1,
        INVALID_NODE_TYPE_ERR: 2
    };
    v.prototype.toString = function() {
        return this.message
    };
    c.prototype = {
        _current: null,
        hasNext: function() {
            return !!this._next
        },
        next: function() {
            this._current = this._next;
            this._next = this.nodes[++this._position];
            return this._current
        },
        detach: function() {
            this._current = this._next = this.nodes = null
        }
    };
    var $ = [1, 3, 4, 5, 7, 8, 10], ba = [2, 9, 11], aa = [1, 3, 4, 5, 7, 8, 10, 11], b = [1, 3, 4, 5, 7, 8], d = g.getRootContainer, h = f([9, 11]), D = f([5, 6, 10, 12]), G = f([6, 10, 12]), P = document.createElement("style"), X = false;
    try {
        P.innerHTML = "<b>x</b>";
        X = P.firstChild.nodeType == 3
    } catch (ta) {}
    l.features.htmlParsingConforms = X;
    var ka = ["startContainer", "startOffset", "endContainer", "endOffset",
    "collapsed", "commonAncestorContainer"], ia = 0, la = 1, ra = 2, ma = 3, na = 0, oa = 1, pa = 2, ja = 3;
    ca.prototype = {
        attachListener: function(a, e) {
            this._listeners[a].push(e)
        },
        compareBoundaryPoints: function(a, e) {
            J(this);
            u(this.startContainer, e.startContainer);
            var j = a == ma || a == ia ? "start": "end", o = a == la || a == ia ? "start": "end";
            return g.comparePoints(this[j + "Container"], this[j + "Offset"], e[o + "Container"], e[o + "Offset"])
        },
        insertNode: function(a) {
            J(this);
            L(a, aa);
            w(this.startContainer);
            if (g.isAncestorOf(a, this.startContainer, true))
                throw new S("HIERARCHY_REQUEST_ERR");
            this.setStartBefore(N(a, this.startContainer, this.startOffset))
        },
        cloneContents: function() {
            J(this);
            var a, e;
            if (this.collapsed)
                return H(this).createDocumentFragment();
            else {
                if (this.startContainer === this.endContainer && g.isCharacterDataNode(this.startContainer)) {
                    a = this.startContainer.cloneNode(true);
                    a.data = a.data.slice(this.startOffset, this.endOffset);
                    e = H(this).createDocumentFragment();
                    e.appendChild(a);
                    return e
                } else {
                    e = new q(this, true);
                    a = O(e);
                    e.detach()
                }
                return a
            }
        },
        canSurroundContents: function() {
            J(this);
            w(this.startContainer);
            w(this.endContainer);
            var a = new q(this, true), e = a._first && K(a._first, this) || a._last && K(a._last, this);
            a.detach();
            return !e
        },
        surroundContents: function(a) {
            L(a, b);
            if (!this.canSurroundContents())
                throw new v("BAD_BOUNDARYPOINTS_ERR");
            var e = this.extractContents();
            if (a.hasChildNodes())
                for (; a.lastChild;)
                    a.removeChild(a.lastChild);
            N(a, this.startContainer, this.startOffset);
            a.appendChild(e);
            this.selectNode(a)
        },
        cloneRange: function() {
            J(this);
            for (var a = new M(H(this)), e = ka.length, j; e--;) {
                j = ka[e];
                a[j] = this[j]
            }
            return a
        },
        toString: function() {
            J(this);
            var a = this.startContainer;
            if (a === this.endContainer && g.isCharacterDataNode(a))
                return a.nodeType == 3 || a.nodeType == 4 ? a.data.slice(this.startOffset, this.endOffset) : "";
            else {
                var e = [];
                a = new q(this, true);
                i(a, function(j) {
                    if (j.nodeType == 3 || j.nodeType == 4)
                        e.push(j.data)
                });
                a.detach();
                return e.join("")
            }
        },
        compareNode: function(a) {
            J(this);
            var e = a.parentNode, j = g.getNodeIndex(a);
            if (!e)
                throw new S("NOT_FOUND_ERR");
            a = this.comparePoint(e, j);
            e = this.comparePoint(e, j + 1);
            return a < 0 ? e > 0 ? pa : na : e > 0 ?
            oa : ja
        },
        comparePoint: function(a, e) {
            J(this);
            B(a, "HIERARCHY_REQUEST_ERR");
            u(a, this.startContainer);
            if (g.comparePoints(a, e, this.startContainer, this.startOffset) < 0)
                return - 1;
            else if (g.comparePoints(a, e, this.endContainer, this.endOffset) > 0)
                return 1;
            return 0
        },
        createContextualFragment: X ? function(a) {
            var e = this.startContainer, j = g.getDocument(e);
            if (!e)
                throw new S("INVALID_STATE_ERR");
            var o = null;
            if (e.nodeType == 1)
                o = e;
            else if (g.isCharacterDataNode(e))
                o = g.parentElement(e);
            o = o === null || o.nodeName == "HTML" && g.isHtmlNamespace(g.getDocument(o).documentElement) &&
            g.isHtmlNamespace(o) ? j.createElement("body") : o.cloneNode(false);
            o.innerHTML = a;
            return g.fragmentFromNodeChildren(o)
        }
        : function(a) {
            r(this);
            var e = H(this).createElement("body");
            e.innerHTML = a;
            return g.fragmentFromNodeChildren(e)
        },
        toHtml: function() {
            J(this);
            var a = H(this).createElement("div");
            a.appendChild(this.cloneContents());
            return a.innerHTML
        },
        intersectsNode: function(a, e) {
            J(this);
            B(a, "NOT_FOUND_ERR");
            if (g.getDocument(a) !== H(this))
                return false;
            var j = a.parentNode, o = g.getNodeIndex(a);
            B(j, "NOT_FOUND_ERR");
            var E = g.comparePoints(j, o, this.endContainer, this.endOffset);
            j = g.comparePoints(j, o + 1, this.startContainer, this.startOffset);
            return e ? E <= 0 && j >= 0 : E < 0 && j > 0
        },
        isPointInRange: function(a, e) {
            J(this);
            B(a, "HIERARCHY_REQUEST_ERR");
            u(a, this.startContainer);
            return g.comparePoints(a, e, this.startContainer, this.startOffset) >= 0 && g.comparePoints(a, e, this.endContainer, this.endOffset) <= 0
        },
        intersectsRange: function(a, e) {
            J(this);
            if (H(a) != H(this))
                throw new S("WRONG_DOCUMENT_ERR");
            var j = g.comparePoints(this.startContainer,
            this.startOffset, a.endContainer, a.endOffset), o = g.comparePoints(this.endContainer, this.endOffset, a.startContainer, a.startOffset);
            return e ? j <= 0 && o >= 0 : j < 0 && o > 0
        },
        intersection: function(a) {
            if (this.intersectsRange(a)) {
                var e = g.comparePoints(this.startContainer, this.startOffset, a.startContainer, a.startOffset), j = g.comparePoints(this.endContainer, this.endOffset, a.endContainer, a.endOffset), o = this.cloneRange();
                e==-1 && o.setStart(a.startContainer, a.startOffset);
                j == 1 && o.setEnd(a.endContainer, a.endOffset);
                return o
            }
            return null
        },
        union: function(a) {
            if (this.intersectsRange(a, true)) {
                var e = this.cloneRange();
                g.comparePoints(a.startContainer, a.startOffset, this.startContainer, this.startOffset)==-1 && e.setStart(a.startContainer, a.startOffset);
                g.comparePoints(a.endContainer, a.endOffset, this.endContainer, this.endOffset) == 1 && e.setEnd(a.endContainer, a.endOffset);
                return e
            } else 
                throw new v("Ranges do not intersect");
        },
        containsNode: function(a, e) {
            return e ? this.intersectsNode(a, false) : this.compareNode(a) == ja
        },
        containsNodeContents: function(a) {
            return this.comparePoint(a,
            0) >= 0 && this.comparePoint(a, g.getNodeLength(a)) <= 0
        },
        containsRange: function(a) {
            return this.intersection(a).equals(a)
        },
        containsNodeText: function(a) {
            var e = this.cloneRange();
            e.selectNode(a);
            var j = e.getNodes([3]);
            if (j.length > 0) {
                e.setStart(j[0], 0);
                a = j.pop();
                e.setEnd(a, a.length);
                a = this.containsRange(e);
                e.detach();
                return a
            } else 
                return this.containsNodeContents(a)
        },
        createNodeIterator: function(a, e) {
            J(this);
            return new c(this, a, e)
        },
        getNodes: function(a, e) {
            J(this);
            return x(this, a, e)
        },
        getDocument: function() {
            return H(this)
        },
        collapseBefore: function(a) {
            r(this);
            this.setEndBefore(a);
            this.collapse(false)
        },
        collapseAfter: function(a) {
            r(this);
            this.setStartAfter(a);
            this.collapse(true)
        },
        getName: function() {
            return "DomRange"
        },
        equals: function(a) {
            return M.rangesEqual(this, a)
        },
        isValid: function() {
            return V(this)
        },
        inspect: function() {
            return A(this)
        }
    };
    fa(M, ha, function(a) {
        r(a);
        a.startContainer = a.startOffset = a.endContainer = a.endOffset = null;
        a.collapsed = a.commonAncestorContainer = null;
        I(a, "detach", null);
        a._listeners = null
    });
    l.rangePrototype = ca.prototype;
    M.rangeProperties = ka;
    M.RangeIterator = q;
    M.copyComparisonConstants = W;
    M.createPrototypeRange = fa;
    M.inspect = A;
    M.getRangeDocument = H;
    M.rangesEqual = function(a, e) {
        return a.startContainer === e.startContainer && a.startOffset === e.startOffset && a.endContainer === e.endContainer && a.endOffset === e.endOffset
    };
    l.DomRange = M;
    l.RangeException = v
});
rangy.createModule("WrappedRange", function(l) {
    function K(i, n, t, x) {
        var A = i.duplicate();
        A.collapse(t);
        var q = A.parentElement();
        z.isAncestorOf(n, q, true) || (q = n);
        if (!q.canHaveHTML)
            return new C(q.parentNode, z.getNodeIndex(q));
        n = z.getDocument(q).createElement("span");
        var v, c = t ? "StartToStart": "StartToEnd";
        do {
            q.insertBefore(n, n.previousSibling);
            A.moveToElementText(n)
        }
        while ((v = A.compareEndPoints(c, i)) > 0 && n.previousSibling);
        c = n.nextSibling;
        if (v==-1 && c && z.isCharacterDataNode(c)) {
            A.setEndPoint(t ? "EndToStart" : "EndToEnd",
            i);
            if (/[\r\n]/.test(c.data)) {
                q = A.duplicate();
                t = q.text.replace(/\r\n/g, "\r").length;
                for (t = q.moveStart("character", t); q.compareEndPoints("StartToEnd", q)==-1;) {
                    t++;
                    q.moveStart("character", 1)
                }
            } else 
                t = A.text.length;
            q = new C(c, t)
        } else {
            c = (x ||!t) && n.previousSibling;
            q = (t = (x || t) && n.nextSibling) && z.isCharacterDataNode(t) ? new C(t, 0) : c && z.isCharacterDataNode(c) ? new C(c, c.length) : new C(q, z.getNodeIndex(n))
        }
        n.parentNode.removeChild(n);
        return q
    }
    function H(i, n) {
        var t, x, A = i.offset, q = z.getDocument(i.node), v = q.body.createTextRange(),
        c = z.isCharacterDataNode(i.node);
        if (c) {
            t = i.node;
            x = t.parentNode
        } else {
            t = i.node.childNodes;
            t = A < t.length ? t[A] : null;
            x = i.node
        }
        q = q.createElement("span");
        q.innerHTML = "&#feff;";
        t ? x.insertBefore(q, t) : x.appendChild(q);
        v.moveToElementText(q);
        v.collapse(!n);
        x.removeChild(q);
        if (c)
            v[n ? "moveStart": "moveEnd"]("character", A);
        return v
    }
    l.requireModules(["DomUtil", "DomRange"]);
    var I, z = l.dom, C = z.DomPosition, N = l.DomRange;
    if (l.features.implementsDomRange && (!l.features.implementsTextRange ||!l.config.preferTextRange)) {
        (function() {
            function i(f) {
                for (var k =
                t.length, r; k--;) {
                    r = t[k];
                    f[r] = f.nativeRange[r]
                }
            }
            var n, t = N.rangeProperties, x, A;
            I = function(f) {
                if (!f)
                    throw Error("Range must be specified");
                this.nativeRange = f;
                i(this)
            };
            N.createPrototypeRange(I, function(f, k, r, L, p) {
                var u = f.endContainer !== L || f.endOffset != p;
                if (f.startContainer !== k || f.startOffset != r || u) {
                    f.setEnd(L, p);
                    f.setStart(k, r)
                }
            }, function(f) {
                f.nativeRange.detach();
                f.detached = true;
                for (var k = t.length, r; k--;) {
                    r = t[k];
                    f[r] = null
                }
            });
            n = I.prototype;
            n.selectNode = function(f) {
                this.nativeRange.selectNode(f);
                i(this)
            };
            n.deleteContents = function() {
                this.nativeRange.deleteContents();
                i(this)
            };
            n.extractContents = function() {
                var f = this.nativeRange.extractContents();
                i(this);
                return f
            };
            n.cloneContents = function() {
                return this.nativeRange.cloneContents()
            };
            n.surroundContents = function(f) {
                this.nativeRange.surroundContents(f);
                i(this)
            };
            n.collapse = function(f) {
                this.nativeRange.collapse(f);
                i(this)
            };
            n.cloneRange = function() {
                return new I(this.nativeRange.cloneRange())
            };
            n.refresh = function() {
                i(this)
            };
            n.toString = function() {
                return this.nativeRange.toString()
            };
            var q = document.createTextNode("test");
            z.getBody(document).appendChild(q);
            var v = document.createRange();
            v.setStart(q, 0);
            v.setEnd(q, 0);
            try {
                v.setStart(q, 1);
                x = true;
                n.setStart = function(f, k) {
                    this.nativeRange.setStart(f, k);
                    i(this)
                };
                n.setEnd = function(f, k) {
                    this.nativeRange.setEnd(f, k);
                    i(this)
                };
                A = function(f) {
                    return function(k) {
                        this.nativeRange[f](k);
                        i(this)
                    }
                }
            } catch (c) {
                x = false;
                n.setStart = function(f, k) {
                    try {
                        this.nativeRange.setStart(f, k)
                    } catch (r) {
                        this.nativeRange.setEnd(f, k);
                        this.nativeRange.setStart(f, k)
                    }
                    i(this)
                };
                n.setEnd = function(f, k) {
                    try {
                        this.nativeRange.setEnd(f, k)
                    } catch (r) {
                        this.nativeRange.setStart(f, k);
                        this.nativeRange.setEnd(f, k)
                    }
                    i(this)
                };
                A = function(f, k) {
                    return function(r) {
                        try {
                            this.nativeRange[f](r)
                        } catch (L) {
                            this.nativeRange[k](r);
                            this.nativeRange[f](r)
                        }
                        i(this)
                    }
                }
            }
            n.setStartBefore = A("setStartBefore", "setEndBefore");
            n.setStartAfter = A("setStartAfter", "setEndAfter");
            n.setEndBefore = A("setEndBefore", "setStartBefore");
            n.setEndAfter = A("setEndAfter", "setStartAfter");
            v.selectNodeContents(q);
            n.selectNodeContents =
            v.startContainer == q && v.endContainer == q && v.startOffset == 0 && v.endOffset == q.length ? function(f) {
                this.nativeRange.selectNodeContents(f);
                i(this)
            } : function(f) {
                this.setStart(f, 0);
                this.setEnd(f, N.getEndOffset(f))
            };
            v.selectNodeContents(q);
            v.setEnd(q, 3);
            x = document.createRange();
            x.selectNodeContents(q);
            x.setEnd(q, 4);
            x.setStart(q, 2);
            n.compareBoundaryPoints = v.compareBoundaryPoints(v.START_TO_END, x)==-1 & v.compareBoundaryPoints(v.END_TO_START, x) == 1 ? function(f, k) {
                k = k.nativeRange || k;
                if (f == k.START_TO_END)
                    f = k.END_TO_START;
                else if (f == k.END_TO_START)
                    f = k.START_TO_END;
                return this.nativeRange.compareBoundaryPoints(f, k)
            } : function(f, k) {
                return this.nativeRange.compareBoundaryPoints(f, k.nativeRange || k)
            };
            if (l.util.isHostMethod(v, "createContextualFragment"))
                n.createContextualFragment = function(f) {
                    return this.nativeRange.createContextualFragment(f)
                };
            z.getBody(document).removeChild(q);
            v.detach();
            x.detach()
        })();
        l.createNativeRange = function(i) {
            i = i || document;
            return i.createRange()
        }
    } else if (l.features.implementsTextRange) {
        I = function(i) {
            this.textRange =
            i;
            this.refresh()
        };
        I.prototype = new N(document);
        I.prototype.refresh = function() {
            var i, n, t = this.textRange;
            i = t.parentElement();
            var x = t.duplicate();
            x.collapse(true);
            n = x.parentElement();
            x = t.duplicate();
            x.collapse(false);
            t = x.parentElement();
            n = n == t ? n : z.getCommonAncestor(n, t);
            n = n == i ? n : z.getCommonAncestor(i, n);
            if (this.textRange.compareEndPoints("StartToEnd", this.textRange) == 0)
                n = i = K(this.textRange, n, true, true);
            else {
                i = K(this.textRange, n, true, false);
                n = K(this.textRange, n, false, false)
            }
            this.setStart(i.node, i.offset);
            this.setEnd(n.node, n.offset)
        };
        N.copyComparisonConstants(I);
        var O = function() {
            return this
        }();
        if (typeof O.Range == "undefined")
            O.Range = I;
        l.createNativeRange = function(i) {
            i = i || document;
            return i.body.createTextRange()
        }
    }
    if (l.features.implementsTextRange)
        I.rangeToTextRange = function(i) {
            if (i.collapsed)
                return H(new C(i.startContainer, i.startOffset), true);
            else {
                var n = H(new C(i.startContainer, i.startOffset), true), t = H(new C(i.endContainer, i.endOffset), false);
                i = z.getDocument(i.startContainer).body.createTextRange();
                i.setEndPoint("StartToStart", n);
                i.setEndPoint("EndToEnd", t);
                return i
            }
        };
    I.prototype.getName = function() {
        return "WrappedRange"
    };
    l.WrappedRange = I;
    l.createRange = function(i) {
        i = i || document;
        return new I(l.createNativeRange(i))
    };
    l.createRangyRange = function(i) {
        i = i || document;
        return new N(i)
    };
    l.createIframeRange = function(i) {
        return l.createRange(z.getIframeDocument(i))
    };
    l.createIframeRangyRange = function(i) {
        return l.createRangyRange(z.getIframeDocument(i))
    };
    l.addCreateMissingNativeApiListener(function(i) {
        i = i.document;
        if (typeof i.createRange == "undefined")
            i.createRange = function() {
                return l.createRange(this)
            };
        i = i = null
    })
});
rangy.createModule("WrappedSelection", function(l, K) {
    function H(b) {
        return (b || window).getSelection()
    }
    function I(b) {
        return (b || window).document.selection
    }
    function z(b, d, h) {
        var D = h ? "end": "start";
        h = h ? "start" : "end";
        b.anchorNode = d[D + "Container"];
        b.anchorOffset = d[D + "Offset"];
        b.focusNode = d[h + "Container"];
        b.focusOffset = d[h + "Offset"]
    }
    function C(b) {
        b.anchorNode = b.focusNode = null;
        b.anchorOffset = b.focusOffset = 0;
        b.rangeCount = 0;
        b.isCollapsed = true;
        b._ranges.length = 0
    }
    function N(b) {
        var d;
        if (b instanceof k) {
            d = b._selectionNativeRange;
            if (!d) {
                d = l.createNativeRange(c.getDocument(b.startContainer));
                d.setEnd(b.endContainer, b.endOffset);
                d.setStart(b.startContainer, b.startOffset);
                b._selectionNativeRange = d;
                b.attachListener("detach", function() {
                    this._selectionNativeRange = null
                })
            }
        } else if (b instanceof r)
            d = b.nativeRange;
        else if (l.features.implementsDomRange && b instanceof c.getWindow(b.startContainer).Range)
            d = b;
        return d
    }
    function O(b) {
        var d = b.getNodes(), h;
        a: if (!d.length || d[0].nodeType != 1)
            h = false;
        else {
            h = 1;
            for (var D = d.length; h < D; ++h)
                if (!c.isAncestorOf(d[0],
                d[h])) {
                    h = false;
                    break a
                }
            h = true
        }
        if (!h)
            throw Error("getSingleElementFromRange: range " + b.inspect() + " did not consist of a single element");
        return d[0]
    }
    function i(b, d) {
        var h = new r(d);
        b._ranges = [h];
        z(b, h, false);
        b.rangeCount = 1;
        b.isCollapsed = h.collapsed
    }
    function n(b) {
        b._ranges.length = 0;
        if (b.docSelection.type == "None")
            C(b);
        else {
            var d = b.docSelection.createRange();
            if (d && typeof d.text != "undefined")
                i(b, d);
            else {
                b.rangeCount = d.length;
                for (var h, D = c.getDocument(d.item(0)), G = 0; G < b.rangeCount; ++G) {
                    h = l.createRange(D);
                    h.selectNode(d.item(G));
                    b._ranges.push(h)
                }
                b.isCollapsed = b.rangeCount == 1 && b._ranges[0].collapsed;
                z(b, b._ranges[b.rangeCount - 1], false)
            }
        }
    }
    function t(b, d) {
        var h = b.docSelection.createRange(), D = O(d), G = c.getDocument(h.item(0));
        G = c.getBody(G).createControlRange();
        for (var P = 0, X = h.length; P < X; ++P)
            G.add(h.item(P));
        try {
            G.add(D)
        } catch (ta) {
            throw Error("addRange(): Element within the specified Range could not be added to control selection (does it have layout?)");
        }
        G.select();
        n(b)
    }
    function x(b, d, h) {
        this.nativeSelection =
        b;
        this.docSelection = d;
        this._ranges = [];
        this.win = h;
        this.refresh()
    }
    function A(b, d) {
        var h = c.getDocument(d[0].startContainer);
        h = c.getBody(h).createControlRange();
        for (var D = 0, G; D < rangeCount; ++D) {
            G = O(d[D]);
            try {
                h.add(G)
            } catch (P) {
                throw Error("setRanges(): Element within the one of the specified Ranges could not be added to control selection (does it have layout?)");
            }
        }
        h.select();
        n(b)
    }
    function q(b, d) {
        if (b.anchorNode && c.getDocument(b.anchorNode) !== c.getDocument(d))
            throw new L("WRONG_DOCUMENT_ERR");
    }
    function v(b) {
        var d =
        [], h = new p(b.anchorNode, b.anchorOffset), D = new p(b.focusNode, b.focusOffset), G = typeof b.getName == "function" ? b.getName(): "Selection";
        if (typeof b.rangeCount != "undefined")
            for (var P = 0, X = b.rangeCount; P < X; ++P)
                d[P] = k.inspect(b.getRangeAt(P));
        return "[" + G + "(Ranges: " + d.join(", ") + ")(anchor: " + h.inspect() + ", focus: " + D.inspect() + "]"
    }
    l.requireModules(["DomUtil", "DomRange", "WrappedRange"]);
    l.config.checkSelectionRanges = true;
    var c = l.dom, f = l.util, k = l.DomRange, r = l.WrappedRange, L = l.DOMException, p = c.DomPosition, u, w,
    B = l.util.isHostMethod(window, "getSelection"), V = l.util.isHostObject(document, "selection"), J = V && (!B || l.config.preferTextRange);
    if (J) {
        u = I;
        l.isSelectionValid = function(b) {
            b = (b || window).document;
            var d = b.selection;
            return d.type != "None" || c.getDocument(d.createRange().parentElement()) == b
        }
    } else if (B) {
        u = H;
        l.isSelectionValid = function() {
            return true
        }
    } else 
        K.fail("Neither document.selection or window.getSelection() detected.");
    l.getNativeSelection = u;
    B = u();
    var ca = l.createNativeRange(document), Y = c.getBody(document),
    W = f.areHostObjects(B, f.areHostProperties(B, ["anchorOffset", "focusOffset"]));
    l.features.selectionHasAnchorAndFocus = W;
    var da = f.isHostMethod(B, "extend");
    l.features.selectionHasExtend = da;
    var fa = typeof B.rangeCount == "number";
    l.features.selectionHasRangeCount = fa;
    var ea = false, ha = true;
    f.areHostMethods(B, ["addRange", "getRangeAt", "removeAllRanges"]) && typeof B.rangeCount == "number" && l.features.implementsDomRange && function() {
        var b = document.createElement("iframe");
        b.frameBorder = 0;
        b.style.position = "absolute";
        b.style.left =
        "-10000px";
        Y.appendChild(b);
        var d = c.getIframeDocument(b);
        d.open();
        d.write("<html><head></head><body>12</body></html>");
        d.close();
        var h = c.getIframeWindow(b).getSelection(), D = d.documentElement.lastChild.firstChild;
        d = d.createRange();
        d.setStart(D, 1);
        d.collapse(true);
        h.addRange(d);
        ha = h.rangeCount == 1;
        h.removeAllRanges();
        var G = d.cloneRange();
        d.setStart(D, 0);
        G.setEnd(D, 2);
        h.addRange(d);
        h.addRange(G);
        ea = h.rangeCount == 2;
        d.detach();
        G.detach();
        Y.removeChild(b)
    }();
    l.features.selectionSupportsMultipleRanges = ea;
    l.features.collapsedNonEditableSelectionsSupported = ha;
    var M = false, g;
    if (Y && f.isHostMethod(Y, "createControlRange")) {
        g = Y.createControlRange();
        if (f.areHostProperties(g, ["item", "add"]))
            M = true
    }
    l.features.implementsControlRange = M;
    w = W ? function(b) {
        return b.anchorNode === b.focusNode && b.anchorOffset === b.focusOffset
    } : function(b) {
        return b.rangeCount ? b.getRangeAt(b.rangeCount - 1).collapsed : false
    };
    var Z;
    if (f.isHostMethod(B, "getRangeAt"))
        Z = function(b, d) {
            try {
                return b.getRangeAt(d)
            } catch (h) {
                return null
            }
        };
    else if (W)
        Z =
        function(b) {
            var d = c.getDocument(b.anchorNode);
            d = l.createRange(d);
            d.setStart(b.anchorNode, b.anchorOffset);
            d.setEnd(b.focusNode, b.focusOffset);
            if (d.collapsed !== this.isCollapsed) {
                d.setStart(b.focusNode, b.focusOffset);
                d.setEnd(b.anchorNode, b.anchorOffset)
            }
            return d
        };
    l.getSelection = function(b) {
        b = b || window;
        var d = b._rangySelection, h = u(b), D = V ? I(b): null;
        if (d) {
            d.nativeSelection = h;
            d.docSelection = D;
            d.refresh(b)
        } else {
            d = new x(h, D, b);
            b._rangySelection = d
        }
        return d
    };
    l.getIframeSelection = function(b) {
        return l.getSelection(c.getIframeWindow(b))
    };
    g = x.prototype;
    if (!J && W && f.areHostMethods(B, ["removeAllRanges", "addRange"])) {
        g.removeAllRanges = function() {
            this.nativeSelection.removeAllRanges();
            C(this)
        };
        var S = function(b, d) {
            var h = k.getRangeDocument(d);
            h = l.createRange(h);
            h.collapseToPoint(d.endContainer, d.endOffset);
            b.nativeSelection.addRange(N(h));
            b.nativeSelection.extend(d.startContainer, d.startOffset);
            b.refresh()
        };
        g.addRange = fa ? function(b, d) {
            if (M && V && this.docSelection.type == "Control")
                t(this, b);
            else if (d && da)
                S(this, b);
            else {
                var h;
                if (ea)
                    h = this.rangeCount;
                else {
                    this.removeAllRanges();
                    h = 0
                }
                this.nativeSelection.addRange(N(b));
                this.rangeCount = this.nativeSelection.rangeCount;
                if (this.rangeCount == h + 1) {
                    if (l.config.checkSelectionRanges)
                        if ((h = Z(this.nativeSelection, this.rangeCount - 1))&&!k.rangesEqual(h, b))
                            b = new r(h);
                    this._ranges[this.rangeCount - 1] = b;
                    z(this, b, aa(this.nativeSelection));
                    this.isCollapsed = w(this)
                } else 
                    this.refresh()
            }
        } : function(b, d) {
            if (d && da)
                S(this, b);
            else {
                this.nativeSelection.addRange(N(b));
                this.refresh()
            }
        };
        g.setRanges = function(b) {
            if (M && b.length >
            1)
                A(this, b);
            else {
                this.removeAllRanges();
                for (var d = 0, h = b.length; d < h; ++d)
                    this.addRange(b[d])
            }
        }
    } else if (f.isHostMethod(B, "empty") && f.isHostMethod(ca, "select") && M && J) {
        g.removeAllRanges = function() {
            try {
                this.docSelection.empty();
                if (this.docSelection.type != "None") {
                    var b;
                    if (this.anchorNode)
                        b = c.getDocument(this.anchorNode);
                    else if (this.docSelection.type == "Control") {
                        var d = this.docSelection.createRange();
                        if (d.length)
                            b = c.getDocument(d.item(0)).body.createTextRange()
                        }
                    if (b) {
                        b.body.createTextRange().select();
                        this.docSelection.empty()
                    }
                }
            } catch (h) {}
            C(this)
        };
        g.addRange = function(b) {
            if (this.docSelection.type == "Control")
                t(this, b);
            else {
                r.rangeToTextRange(b).select();
                this._ranges[0] = b;
                this.rangeCount = 1;
                this.isCollapsed = this._ranges[0].collapsed;
                z(this, b, false)
            }
        };
        g.setRanges = function(b) {
            this.removeAllRanges();
            var d = b.length;
            if (d > 1)
                A(this, b);
            else 
                d && this.addRange(b[0])
        }
    } else {
        K.fail("No means of selecting a Range or TextRange was found");
        return false
    }
    g.getRangeAt = function(b) {
        if (b < 0 || b >= this.rangeCount)
            throw new L("INDEX_SIZE_ERR");
        else 
            return this._ranges[b]
    };
    var $;
    if (J)
        $ = function(b) {
            var d;
            if (l.isSelectionValid(b.win))
                d = b.docSelection.createRange();
            else {
                d = c.getBody(b.win.document).createTextRange();
                d.collapse(true)
            }
            if (b.docSelection.type == "Control")
                n(b);
            else 
                d && typeof d.text != "undefined" ? i(b, d) : C(b)
            };
    else if (f.isHostMethod(B, "getRangeAt") && typeof B.rangeCount == "number")
        $ = function(b) {
            if (M && V && b.docSelection.type == "Control")
                n(b);
            else {
                b._ranges.length = b.rangeCount = b.nativeSelection.rangeCount;
                if (b.rangeCount) {
                    for (var d = 0, h = b.rangeCount; d < h; ++d)
                        b._ranges[d] =
                        new l.WrappedRange(b.nativeSelection.getRangeAt(d));
                        z(b, b._ranges[b.rangeCount - 1], aa(b.nativeSelection));
                        b.isCollapsed = w(b)
                    } else 
                        C(b)
            }
        };
    else if (W && typeof B.isCollapsed == "boolean" && typeof ca.collapsed == "boolean" && l.features.implementsDomRange)
        $ = function(b) {
            var d;
            d = b.nativeSelection;
            if (d.anchorNode) {
                d = Z(d, 0);
                b._ranges = [d];
                b.rangeCount = 1;
                d = b.nativeSelection;
                b.anchorNode = d.anchorNode;
                b.anchorOffset = d.anchorOffset;
                b.focusNode = d.focusNode;
                b.focusOffset = d.focusOffset;
                b.isCollapsed = w(b)
            } else 
                C(b)
            };
    else {
        K.fail("No means of obtaining a Range or TextRange from the user's selection was found");
        return false
    }
    g.refresh = function(b) {
        var d = b ? this._ranges.slice(0): null;
        $(this);
        if (b) {
            b = d.length;
            if (b != this._ranges.length)
                return false;
            for (; b--;)
                if (!k.rangesEqual(d[b], this._ranges[b]))
                    return false;
            return true
        }
    };
    var ba = function(b, d) {
        var h = b.getAllRanges(), D = false;
        b.removeAllRanges();
        for (var G = 0, P = h.length; G < P; ++G)
            if (D || d !== h[G])
                b.addRange(h[G]);
            else 
                D = true;
        b.rangeCount || C(b)
    };
    g.removeRange = M ? function(b) {
        if (this.docSelection.type == "Control") {
            var d = this.docSelection.createRange();
            b = O(b);
            var h = c.getDocument(d.item(0));
            h = c.getBody(h).createControlRange();
            for (var D, G = false, P = 0, X = d.length; P < X; ++P) {
                D = d.item(P);
                if (D !== b || G)
                    h.add(d.item(P));
                else 
                    G = true
            }
            h.select();
            n(this)
        } else 
            ba(this, b)
    } : function(b) {
        ba(this, b)
    };
    var aa;
    if (!J && W && l.features.implementsDomRange) {
        aa = function(b) {
            var d = false;
            if (b.anchorNode)
                d = c.comparePoints(b.anchorNode, b.anchorOffset, b.focusNode, b.focusOffset) == 1;
            return d
        };
        g.isBackwards = function() {
            return aa(this)
        }
    } else 
        aa = g.isBackwards = function() {
            return false
        };
    g.toString = function() {
        for (var b = [], d = 0, h = this.rangeCount; d <
        h; ++d)
            b[d] = "" + this._ranges[d];
        return b.join("")
    };
    g.collapse = function(b, d) {
        q(this, b);
        var h = l.createRange(c.getDocument(b));
        h.collapseToPoint(b, d);
        this.removeAllRanges();
        this.addRange(h);
        this.isCollapsed = true
    };
    g.collapseToStart = function() {
        if (this.rangeCount) {
            var b = this._ranges[0];
            this.collapse(b.startContainer, b.startOffset)
        } else 
            throw new L("INVALID_STATE_ERR");
    };
    g.collapseToEnd = function() {
        if (this.rangeCount) {
            var b = this._ranges[this.rangeCount - 1];
            this.collapse(b.endContainer, b.endOffset)
        } else 
            throw new L("INVALID_STATE_ERR");
    };
    g.selectAllChildren = function(b) {
        q(this, b);
        var d = l.createRange(c.getDocument(b));
        d.selectNodeContents(b);
        this.removeAllRanges();
        this.addRange(d)
    };
    g.deleteFromDocument = function() {
        if (M && V && this.docSelection.type == "Control") {
            for (var b = this.docSelection.createRange(), d; b.length;) {
                d = b.item(0);
                b.remove(d);
                d.parentNode.removeChild(d)
            }
            this.refresh()
        } else if (this.rangeCount) {
            b = this.getAllRanges();
            this.removeAllRanges();
            d = 0;
            for (var h = b.length; d < h; ++d)
                b[d].deleteContents();
            this.addRange(b[h - 1])
        }
    };
    g.getAllRanges =
    function() {
        return this._ranges.slice(0)
    };
    g.setSingleRange = function(b) {
        this.setRanges([b])
    };
    g.containsNode = function(b, d) {
        for (var h = 0, D = this._ranges.length; h < D; ++h)
            if (this._ranges[h].containsNode(b, d))
                return true;
        return false
    };
    g.toHtml = function() {
        var b = "";
        if (this.rangeCount) {
            b = k.getRangeDocument(this._ranges[0]).createElement("div");
            for (var d = 0, h = this._ranges.length; d < h; ++d)
                b.appendChild(this._ranges[d].cloneContents());
            b = b.innerHTML
        }
        return b
    };
    g.getName = function() {
        return "WrappedSelection"
    };
    g.inspect =
    function() {
        return v(this)
    };
    g.detach = function() {
        this.win = this.anchorNode = this.focusNode = this.win._rangySelection = null
    };
    x.inspect = v;
    l.Selection = x;
    l.selectionPrototype = g;
    l.addCreateMissingNativeApiListener(function(b) {
        if (typeof b.getSelection == "undefined")
            b.getSelection = function() {
                return l.getSelection(this)
            };
        b = null
    })
});



/*
 CSS Class Applier module for Rangy.
 Adds, removes and toggles CSS classes on Ranges and Selections

 Part of Rangy, a cross-browser JavaScript range and selection library
 http://code.google.com/p/rangy/

 Depends on Rangy core.

 Copyright 2012, Tim Down
 Licensed under the MIT license.
 Version: 1.2.3
 Build date: 26 February 2012
*/
rangy.createModule("CssClassApplier", function(i, v) {
    function r(a, b) {
        return a.className && RegExp("(?:^|\\s)" + b + "(?:\\s|$)").test(a.className)
    }
    function s(a, b) {
        if (a.className)
            r(a, b) || (a.className += " " + b);
        else 
            a.className = b
    }
    function o(a) {
        return a.split(/\s+/).sort().join(" ")
    }
    function w(a, b) {
        return o(a.className) == o(b.className)
    }
    function x(a) {
        for (var b = a.parentNode; a.hasChildNodes();)
            b.insertBefore(a.firstChild, a);
        b.removeChild(a)
    }
    function y(a, b) {
        var c = a.cloneRange();
        c.selectNodeContents(b);
        var d = c.intersection(a);
        d = d ? d.toString() : "";
        c.detach();
        return d != ""
    }
    function z(a) {
        return a.getNodes([3], function(b) {
            return y(a, b)
        })
    }
    function A(a, b) {
        if (a.attributes.length != b.attributes.length)
            return false;
        for (var c = 0, d = a.attributes.length, e, f; c < d; ++c) {
            e = a.attributes[c];
            f = e.name;
            if (f != "class") {
                f = b.attributes.getNamedItem(f);
                if (!f || (e.specified != f.specified))
                    return false;
                if (e.specified && e.nodeValue !== f.nodeValue)
                    return false
            }
        }
        return true
    }
    function B(a, b) {
        for (var c = 0, d = a.attributes.length, e; c < d; ++c) {
            e = a.attributes[c].name;
            if (!(b &&
            h.arrayContains(b, e)) && a.attributes[c].specified && e != "class")
                return true
        }
        return false
    }
    function C(a) {
        var b;
        return a && a.nodeType == 1 && ((b = a.parentNode) && b.nodeType == 9 && b.designMode == "on" || k(a)&&!k(a.parentNode))
    }
    function D(a) {
        return (k(a) || a.nodeType != 1 && k(a.parentNode))&&!C(a)
    }
    function E(a) {
        return a && a.nodeType == 1&&!M.test(p(a, "display"))
    }
    function N(a) {
        if (a.data.length == 0)
            return true;
        if (O.test(a.data))
            return false;
        switch (p(a.parentNode, "whiteSpace")) {
        case "pre":
        case "pre-wrap":
        case "-moz-pre-wrap":
            return false;
        case "pre-line":
            if (/[\r\n]/.test(a.data))
                return false
        }
        return E(a.previousSibling) || E(a.nextSibling)
    }
    function m(a, b, c, d) {
        var e, f = c == 0;
        if (h.isAncestorOf(b, a))
            return a;
        if (h.isCharacterDataNode(b))
            if (c == 0) {
                c = h.getNodeIndex(b);
                b = b.parentNode
            } else if (c == b.length) {
                c = h.getNodeIndex(b) + 1;
                b = b.parentNode
            } else 
                throw v.createError("splitNodeAt should not be called with offset in the middle of a data node (" + c + " in " + b.data);
        var g;
        g = b;
        var j = c;
        g = h.isCharacterDataNode(g) ? j == 0?!!g.previousSibling : j == g.length?!!g.nextSibling :
        true : j > 0 && j < g.childNodes.length;
        if (g) {
            if (!e) {
                e = b.cloneNode(false);
                for (e.id && e.removeAttribute("id"); f = b.childNodes[c];)
                    e.appendChild(f);
                h.insertAfter(e, b)
            }
            return b == a ? e : m(a, e.parentNode, h.getNodeIndex(e), d)
        } else if (a != b) {
            e = b.parentNode;
            b = h.getNodeIndex(b);
            f || b++;
            return m(a, e, b, d)
        }
        return a
    }
    function F(a) {
        var b = a ? "nextSibling": "previousSibling";
        return function(c, d) {
            var e = c.parentNode, f = c[b];
            if (f) {
                if (f && f.nodeType == 3)
                    return f
            } else if (d)
                if ((f = e[b]) && f.nodeType == 1 && e.tagName == f.tagName && w(e, f) && A(e, f) && p(e, 'display') == 'inline' && p(f, 'display') == 'inline')
                    return f[a ?
                    "firstChild": "lastChild"];
            return null
        }
    }
    function t(a) {
        this.firstTextNode = (this.isElementMerge = a.nodeType == 1) ? a.lastChild : a;
        this.textNodes = [this.firstTextNode]
    }
    function q(a, b, c) {
        this.cssClass = a;
        var d, e, f = null;
        if (typeof b == "object" && b !== null) {
            c = b.tagNames;
            f = b.elementProperties;
            for (d = 0; e = P[d++];)
                if (b.hasOwnProperty(e))
                    this[e] = b[e];
            d = b.normalize
        } else 
            d = b;
        this.normalize = typeof d == "undefined" ? true : d;
        this.attrExceptions = [];
        d = document.createElement(this.elementTagName);
        this.elementProperties = {};
        for (var g in f)
            if (f.hasOwnProperty(g)) {
                if (G.hasOwnProperty(g))
                    g =
                    G[g];
                    d[g] = f[g];
                    this.elementProperties[g] = d[g];
                    this.attrExceptions.push(g)
            }
        this.elementSortedClassName = this.elementProperties.hasOwnProperty("className") ? o(this.elementProperties.className + " " + a) : a;
        this.applyToAnyTagName = false;
        a = typeof c;
        if (a == "string")
            if (c == "*")
                this.applyToAnyTagName = true;
            else 
                this.tagNames = c.toLowerCase().replace(/^\s\s*/, "").replace(/\s\s*$/, "").split(/\s*,\s*/);
        else if (a == "object" && typeof c.length == "number") {
            this.tagNames = [];
            d = 0;
            for (a = c.length; d < a; ++d)
                if (c[d] == "*")
                    this.applyToAnyTagName =
                    true;
                else 
                    this.tagNames.push(c[d].toLowerCase())
        } else 
            this.tagNames = [this.elementTagName]
    }
    i.requireModules(["WrappedSelection", "WrappedRange"]);
    var h = i.dom, H = function() {
        function a(b, c, d) {
            return c && d ? " " : ""
        }
        return function(b, c) {
            if (b.className)
                b.className = b.className.replace(RegExp("(?:^|\\s)" + c + "(?:\\s|$)"), a)
        }
    }(), p;
    if (typeof window.getComputedStyle != "undefined")
        p = function(a, b) {
            return h.getWindow(a).getComputedStyle(a, null)[b]
        };
    else if (typeof document.documentElement.currentStyle != "undefined")
        p = function(a,
        b) {
            return a.currentStyle[b]
        };
    else 
        v.fail("No means of obtaining computed style properties found");
    var k;
    (function() {
        k = typeof document.createElement("div").isContentEditable == "boolean" ? function(a) {
            return a && a.nodeType == 1 && a.isContentEditable
        } : function(a) {
            if (!a || a.nodeType != 1 || a.contentEditable == "false")
                return false;
            return a.contentEditable == "true" || k(a.parentNode)
        }
    })();
    var M = /^inline(-block|-table)?$/i, O = /[^\r\n\t\f \u200B]/, Q = F(false), R = F(true);
    t.prototype = {
        doMerge: function() {
            for (var a = [], b, c, d = 0,
            e = this.textNodes.length; d < e; ++d) {
                b = this.textNodes[d];
                c = b.parentNode;
                a[d] = b.data;
                if (d) {
                    c.removeChild(b);
                    c.hasChildNodes() || c.parentNode.removeChild(c)
                }
            }
            return this.firstTextNode.data = a = a.join("")
        },
        getLength: function() {
            for (var a = this.textNodes.length, b = 0; a--;)
                b += this.textNodes[a].length;
            return b
        },
        toString: function() {
            for (var a = [], b = 0, c = this.textNodes.length; b < c; ++b)
                a[b] = "'" + this.textNodes[b].data + "'";
            return "[Merge(" + a.join(",") + ")]"
        }
    };
    var P = ["elementTagName", "ignoreWhiteSpace", "applyToEditableOnly"],
    G = {
        "class": "className"
    };
    q.prototype = {
        elementTagName: "span",
        elementProperties: {},
        ignoreWhiteSpace: true,
        applyToEditableOnly: false,
        hasClass: function(a) {
            return a.nodeType == 1 && h.arrayContains(this.tagNames, a.tagName.toLowerCase()) && r(a, this.cssClass)
        },
        getSelfOrAncestorWithClass: function(a) {
            for (; a;) {
                if (this.hasClass(a, this.cssClass))
                    return a;
                a = a.parentNode
            }
            return null
        },
        isModifiable: function(a) {
            return !this.applyToEditableOnly || D(a)
        },
        isIgnorableWhiteSpaceNode: function(a) {
            return this.ignoreWhiteSpace && a &&
            a.nodeType == 3 && N(a)
        },
        postApply: function(a, b, c) {
            if (a.length == 0)
                return;
            for (var d = a[0], e = a[a.length - 1], f = [], g, j = d, I = e, J = 0, K = e.length, n, L, l = 0, u = a.length; l < u; ++l) {
                n = a[l];
                if (L = Q(n, !c)) {
                    if (!g) {
                        g = new t(L);
                        f.push(g)
                    }
                    g.textNodes.push(n);
                    if (n === d) {
                        j = g.firstTextNode;
                        J = j.length
                    }
                    if (n === e) {
                        I = g.firstTextNode;
                        K = g.getLength()
                    }
                } else 
                    g = null
            }
            if (a = R(e, !c)) {
                if (!g) {
                    g = new t(e);
                    f.push(g)
                }
                g.textNodes.push(a)
            }
            if (f.length) {
                l = 0;
                for (u = f.length; l < u; ++l)
                    f[l].doMerge();
                b.setStart(j, J);
                b.setEnd(I, K)
            }
        },
        createContainer: function(a) {
            a = a.createElement(this.elementTagName);
            i.util.extend(a, this.elementProperties);
            s(a, this.cssClass);
            return a
        },
        applyToTextNode: function(a) {
            var b = a.parentNode;
            if (b.childNodes.length == 1 && h.arrayContains(this.tagNames, b.tagName.toLowerCase()))
                s(b, this.cssClass);
            else {
                b = this.createContainer(h.getDocument(a));
                a.parentNode.insertBefore(b, a);
                b.appendChild(a)
            }
        },
        isRemovable: function(a) {
            var b;
            if (b = a.tagName.toLowerCase() == this.elementTagName) {
                if (b = o(a.className) == this.elementSortedClassName) {
                    var c;
                    a:
                    {
                        b = this.elementProperties;
                        for (c in b)
                            if (b.hasOwnProperty(c) &&
                            a[c] !== b[c]) {
                                c = false;
                                break a
                            }
                        c = true
                    }
                    b = c&&!B(a, this.attrExceptions) && this.isModifiable(a)
                }
                b = b
            }
            return b
        },
        undoToTextNode: function(a, b, c) {
            if (!b.containsNode(c)) {
                a = b.cloneRange();
                a.selectNode(c);
                if (a.isPointInRange(b.endContainer, b.endOffset)) {
                    m(c, b.endContainer, b.endOffset, [b]);
                    b.setEndAfter(c)
                }
                if (a.isPointInRange(b.startContainer, b.startOffset))
                    c = m(c, b.startContainer, b.startOffset, [b])
            }
            this.isRemovable(c) ? x(c) : H(c, this.cssClass)
        },
        applyToRange: function(a) {
            a.splitBoundaries();
            var b = z(a);
            if (b.length) {
                for (var c,
                d = 0, e = b.length; d < e; ++d) {
                    c = b[d];
                    !this.isIgnorableWhiteSpaceNode(c)&&!this.getSelfOrAncestorWithClass(c) && this.isModifiable(c) && this.applyToTextNode(c)
                }
                a.setStart(b[0], 0);
                c = b[b.length - 1];
                a.setEnd(c, c.length);
                this.normalize && this.postApply(b, a, false)
            }
        },
        applyToSelection: function(a) {
            a = a || window;
            a = i.getSelection(a);
            var b, c = a.getAllRanges();
            a.removeAllRanges();
            for (var d = c.length; d--;) {
                b = c[d];
                this.applyToRange(b);
                a.addRange(b)
            }
        },
        undoToRange: function(a) {
            a.splitBoundaries();
            var b = z(a), c, d, e = b[b.length - 1];
            if (b.length) {
                for (var f =
                0, g = b.length; f < g; ++f) {
                    c = b[f];
                    (d = this.getSelfOrAncestorWithClass(c)) && this.isModifiable(c) && this.undoToTextNode(c, a, d);
                    a.setStart(b[0], 0);
                    a.setEnd(e, e.length)
                }
                this.normalize && this.postApply(b, a, true)
            }
        },
        undoToSelection: function(a) {
            a = a || window;
            a = i.getSelection(a);
            var b = a.getAllRanges(), c;
            a.removeAllRanges();
            for (var d = 0, e = b.length; d < e; ++d) {
                c = b[d];
                this.undoToRange(c);
                a.addRange(c)
            }
        },
        getTextSelectedByRange: function(a, b) {
            var c = b.cloneRange();
            c.selectNodeContents(a);
            var d = c.intersection(b);
            d = d ? d.toString() :
            "";
            c.detach();
            return d
        },
        isAppliedToRange: function(a) {
            if (a.collapsed)
                return !!this.getSelfOrAncestorWithClass(a.commonAncestorContainer);
            else {
                for (var b = a.getNodes([3]), c = 0, d; d = b[c++];)
                    if (!this.isIgnorableWhiteSpaceNode(d) && y(a, d) && this.isModifiable(d)&&!this.getSelfOrAncestorWithClass(d))
                        return false;
                return true
            }
        },
        isAppliedToSelection: function(a) {
            a = a || window;
            a = i.getSelection(a).getAllRanges();
            for (var b = a.length; b--;)
                if (!this.isAppliedToRange(a[b]))
                    return false;
            return true
        },
        toggleRange: function(a) {
            this.isAppliedToRange(a) ?
            this.undoToRange(a) : this.applyToRange(a)
        },
        toggleSelection: function(a) {
            this.isAppliedToSelection(a) ? this.undoToSelection(a) : this.applyToSelection(a)
        },
        detach: function() {}
    };
    q.util = {
        hasClass: r,
        addClass: s,
        removeClass: H,
        hasSameClasses: w,
        replaceWithOwnChildren: x,
        elementsHaveSameNonClassAttributes: A,
        elementHasNonClassAttributes: B,
        splitNodeAt: m,
        isEditableElement: k,
        isEditingHost: C,
        isEditable: D
    };
    i.CssClassApplier = q;
    i.createCssClassApplier = function(a, b, c) {
        return new q(a, b, c)
    }
});




/*
 Selection save and restore module for Rangy.
 Saves and restores user selections using marker invisible elements in the DOM.

 Part of Rangy, a cross-browser JavaScript range and selection library
 http://code.google.com/p/rangy/

 Depends on Rangy core.

 Copyright 2012, Tim Down
 Licensed under the MIT license.
 Version: 1.2.3
 Build date: 26 February 2012
*/
rangy.createModule("SaveRestore", function(h, m) {
    function n(a, g) {
        var e = "selectionBoundary_" + + new Date + "_" + ("" + Math.random()).slice(2), c, f = p.getDocument(a.startContainer), d = a.cloneRange();
        d.collapse(g);
        c = f.createElement("span");
        c.id = e;
        c.style.lineHeight = "0";
        c.style.display = "none";
        c.className = "rangySelectionBoundary";
        c.appendChild(f.createTextNode(q));
        d.insertNode(c);
        d.detach();
        return c
    }
    function o(a, g, e, c) {
        if (a = (a || document).getElementById(e)) {
            g[c ? "setStartBefore": "setEndBefore"](a);
            a.parentNode.removeChild(a)
        } else 
            m.warn("Marker element has been removed. Cannot restore selection.")
    }
    function r(a, g) {
        return g.compareBoundaryPoints(a.START_TO_START, a)
    }
    function k(a, g) {
        var e = (a || document).getElementById(g);
        e && e.parentNode.removeChild(e)
    }
    h.requireModules(["DomUtil", "DomRange", "WrappedRange"]);
    var p = h.dom, q = "\ufeff";
    h.saveSelection = function(a) {
        a = a || window;
        var g = a.document;
        if (h.isSelectionValid(a)) {
            var e = h.getSelection(a), c = e.getAllRanges(), f = [], d, j;
            c.sort(r);
            for (var b = 0, i = c.length; b < i; ++b) {
                d = c[b];
                if (d.collapsed) {
                    j = n(d, false);
                    f.push({
                        markerId: j.id,
                        collapsed: true
                    })
                } else {
                    j = n(d, false);
                    d = n(d, true);
                    f[b] = {
                        startMarkerId: d.id,
                        endMarkerId: j.id,
                        collapsed: false,
                        backwards: c.length == 1 && e.isBackwards()
                    }
                }
            }
            for (b = i - 1; b >= 0; --b) {
                d = c[b];
                if (d.collapsed)
                    d.collapseBefore((g || document).getElementById(f[b].markerId));
                else {
                    d.setEndBefore((g || document).getElementById(f[b].endMarkerId));
                    d.setStartAfter((g || document).getElementById(f[b].startMarkerId))
                }
            }
            e.setRanges(c);
            return {
                win: a,
                doc: g,
                rangeInfos: f,
                restored: false
            }
        } else 
            m.warn("Cannot save selection. This usually happens when the selection is collapsed and the selection document has lost focus.")
    };
    h.restoreSelection = function(a, g) {
        if (!a.restored) {
            for (var e = a.rangeInfos, c = h.getSelection(a.win), f = [], d = e.length, j = d - 1, b, i; j >= 0; --j) {
                b = e[j];
                i = h.createRange(a.doc);
                if (b.collapsed)
                    if (b = (a.doc || document).getElementById(b.markerId)) {
                        b.style.display = "inline";
                        var l = b.previousSibling;
                        if (l && l.nodeType == 3) {
                            b.parentNode.removeChild(b);
                            i.collapseToPoint(l, l.length)
                        } else {
                            i.collapseBefore(b);
                            b.parentNode.removeChild(b)
                        }
                    } else 
                        m.warn("Marker element has been removed. Cannot restore selection.");
                else {
                    o(a.doc, i, b.startMarkerId,
                    true);
                    o(a.doc, i, b.endMarkerId, false)
                }
                d == 1 && i.normalizeBoundaries();
                f[j] = i
            }
            if (d == 1 && g && h.features.selectionHasExtend && e[0].backwards) {
                c.removeAllRanges();
                c.addRange(f[0], true)
            } else 
                c.setRanges(f);
            a.restored = true
        }
    };
    h.removeMarkerElement = k;
    h.removeMarkers = function(a) {
        for (var g = a.rangeInfos, e = 0, c = g.length, f; e < c; ++e) {
            f = g[e];
            if (f.collapsed)
                k(a.doc, f.markerId);
            else {
                k(a.doc, f.startMarkerId);
                k(a.doc, f.endMarkerId)
            }
        }
    }
});




/*
 Serializer module for Rangy.
 Serializes Ranges and Selections. An example use would be to store a user's selection on a particular page in a
 cookie or local storage and restore it on the user's next visit to the same page.

 Part of Rangy, a cross-browser JavaScript range and selection library
 http://code.google.com/p/rangy/

 Depends on Rangy core.

 Copyright 2012, Tim Down
 Licensed under the MIT license.
 Version: 1.2.3
 Build date: 26 February 2012
*/
rangy.createModule("Serializer", function(g, n) {
    function o(c, a) {
        a = a || [];
        var b = c.nodeType, e = c.childNodes, d = e.length, f = [b, c.nodeName, d].join(":"), h = "", k = "";
        switch (b) {
        case 3:
            h = c.nodeValue.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            break;
        case 8:
            h = "<!--" + c.nodeValue.replace(/</g, "&lt;").replace(/>/g, "&gt;") + "--\>";
            break;
        default:
            h = "<" + f + ">";
            k = "</>";
            break
        }
        h && a.push(h);
        for (b = 0; b < d; ++b)
            o(e[b], a);
        k && a.push(k);
        return a
    }
    function j(c) {
        c = o(c).join("");
        return u(c).toString(16)
    }
    function l(c, a, b) {
        var e = [], d = c;
        for (b =
        b || i.getDocument(c).documentElement; d && d != b;) {
            e.push(i.getNodeIndex(d, true));
            d = d.parentNode
        }
        return e.join("/") + ":" + a
    }
    function m(c, a, b) {
        if (a)
            b || i.getDocument(a);
        else {
            b = b || document;
            a = b.documentElement
        }
        c = c.split(":");
        a = a;
        b = c[0] ? c[0].split("/") : [];
        for (var e = b.length, d; e--;) {
            d = parseInt(b[e], 10);
            if (d < a.childNodes.length)
                a = a.childNodes[parseInt(b[e], 10)];
            else 
                throw n.createError("deserializePosition failed: node " + i.inspectNode(a) + " has no child with index " + d + ", " + e);
        }
        return new i.DomPosition(a, parseInt(c[1],
        10))
    }
    function p(c, a, b) {
        b = b || g.DomRange.getRangeDocument(c).documentElement;
        if (!i.isAncestorOf(b, c.commonAncestorContainer, true))
            throw Error("serializeRange: range is not wholly contained within specified root node");
        c = l(c.startContainer, c.startOffset, b) + "," + l(c.endContainer, c.endOffset, b);
        a || (c += "{" + j(b) + "}");
        return c
    }
    function q(c, a, b) {
        if (a)
            b = b || i.getDocument(a);
        else {
            b = b || document;
            a = b.documentElement
        }
        c = /^([^,]+),([^,\{]+)({([^}]+)})?$/.exec(c);
        var e = c[4], d = j(a);
        if (e && e !== j(a))
            throw Error("deserializeRange: checksums of serialized range root node (" +
            e + ") and target root node (" + d + ") do not match");
        e = m(c[1], a, b);
        a = m(c[2], a, b);
        b = g.createRange(b);
        b.setStart(e.node, e.offset);
        b.setEnd(a.node, a.offset);
        return b
    }
    function r(c, a, b) {
        if (a)
            b || i.getDocument(a);
        else {
            b = b || document;
            a = b.documentElement
        }
        c = /^([^,]+),([^,]+)({([^}]+)})?$/.exec(c)[3];
        return !c || c === j(a)
    }
    function s(c, a, b) {
        c = c || g.getSelection();
        c = c.getAllRanges();
        for (var e = [], d = 0, f = c.length; d < f; ++d)
            e[d] = p(c[d], a, b);
        return e.join("|")
    }
    function t(c, a, b) {
        if (a)
            b = b || i.getWindow(a);
        else {
            b = b || window;
            a = b.document.documentElement
        }
        c =
        c.split("|");
        for (var e = g.getSelection(b), d = [], f = 0, h = c.length; f < h; ++f)
            d[f] = q(c[f], a, b.document);
        e.setRanges(d);
        return e
    }
    g.requireModules(["WrappedSelection", "WrappedRange"]);
    if (typeof encodeURIComponent == "undefined" || typeof decodeURIComponent == "undefined")
        n.fail("Global object is missing encodeURIComponent and/or decodeURIComponent method");
    var u = function() {
        var c = null;
        return function(a) {
            for (var b = [], e = 0, d = a.length, f; e < d; ++e) {
                f = a.charCodeAt(e);
                if (f < 128)
                    b.push(f);
                else 
                    f < 2048 ? b.push(f>>6 | 192, f & 63 | 128) :
                    b.push(f>>12 | 224, f>>6 & 63 | 128, f & 63 | 128)
            }
            a =- 1;
            if (!c) {
                e = [];
                d = 0;
                for (var h; d < 256; ++d) {
                    h = d;
                    for (f = 8; f--;)
                        if ((h & 1) == 1)
                            h = h>>>1^3988292384;
                        else 
                            h>>>=1;
                    e[d] = h>>>0
                }
                c = e
            }
            e = c;
            d = 0;
            for (f = b.length; d < f; ++d) {
                h = (a^b[d]) & 255;
                a = a>>>8^e[h]
            }
            return (a^-1)>>>0
        }
    }(), i = g.dom;
    g.serializePosition = l;
    g.deserializePosition = m;
    g.serializeRange = p;
    g.deserializeRange = q;
    g.canDeserializeRange = r;
    g.serializeSelection = s;
    g.deserializeSelection = t;
    g.canDeserializeSelection = function(c, a, b) {
        var e;
        if (a)
            e = b ? b.document : i.getDocument(a);
        else {
            b = b || window;
            a = b.document.documentElement
        }
        c = c.split("|");
        b = 0;
        for (var d = c.length; b < d; ++b)
            if (!r(c[b], a, e))
                return false;
        return true
    };
    g.restoreSelectionFromCookie = function(c) {
        c = c || window;
        var a;
        a: {
            a = c.document.cookie.split(/[;,]/);
            for (var b = 0, e = a.length, d; b < e; ++b) {
                d = a[b].split("=");
                if (d[0].replace(/^\s+/, "") == "rangySerializedSelection")
                    if (d = d[1]) {
                        a = decodeURIComponent(d.replace(/\s+$/, ""));
                        break a
                    }
            }
            a = null
        }
        a && t(a, c.doc)
    };
    g.saveSelectionCookie = function(c, a) {
        c = c || window;
        a = typeof a == "object" ? a : {};
        var b = a.expires ? ";expires=" +
        a.expires.toUTCString(): "", e = a.path ? ";path=" + a.path: "", d = a.domain ? ";domain=" + a.domain: "", f = a.secure ? ";secure": "", h = s(g.getSelection(c));
        c.document.cookie = encodeURIComponent("rangySerializedSelection") + "=" + encodeURIComponent(h) + b + e + d + f
    };
    g.getElementChecksum = j
});
/* jshint ignore:end */
;
define("internal/sitebuilder/builder/rangy", function() {});

/*
 * jQuery Hotkeys Plugin
 * Copyright 2010, John Resig
 * Dual licensed under the MIT or GPL Version 2 licenses.
 *
 * Based upon the plugin by Tzury Bar Yochay:
 * http://github.com/tzuryby/hotkeys
 *
 * Original idea by:
 * Binny V A, http://www.openjs.com/scripts/events/keyboard_shortcuts/
*/
(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define('jquery.hotkeys', ['jquery'], factory);
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function ($) {
    $.hotkeys = {
        version: "0.8",

        specialKeys: {
            8: "backspace",
            9: "tab",
            13: "return",
            16: "shift",
            17: "ctrl",
            18: "alt",
            19: "pause",
            20: "capslock",
            27: "esc",
            32: "space",
            33: "pageup",
            34: "pagedown",
            35: "end",
            36: "home",
            37: "left",
            38: "up",
            39: "right",
            40: "down",
            45: "insert",
            46: "del",
            96: "0",
            97: "1",
            98: "2",
            99: "3",
            100: "4",
            101: "5",
            102: "6",
            103: "7",
            104: "8",
            105: "9",
            106: "*",
            107: "+",
            109: "-",
            110: ".",
            111 : "/",
            112: "f1",
            113: "f2",
            114: "f3",
            115: "f4",
            116: "f5",
            117: "f6",
            118: "f7",
            119: "f8",
            120: "f9",
            121: "f10",
            122: "f11",
            123: "f12",
            144: "numlock",
            145: "scroll",
            191: "/",
            224: "meta"
        },

        shiftNums: {
            "`": "~",
            "1": "!",
            "2": "@",
            "3": "#",
            "4": "$",
            "5": "%",
            "6": "^",
            "7": "&",
            "8": "*",
            "9": "(",
            "0": ")",
            "-": "_",
            "=": "+",
            ";": ": ",
            "'": "\"",
            ",": "<",
            ".": ">",
            "/": "?",
            "\\": "|"
        }
    };

    function keyHandler( handleObj ) {
        // Only care when a possible input has been specified
        if ( typeof handleObj.data !== "string" ) {
            return;
        }

        var origHandler = handleObj.handler,
        keys = handleObj.data.toLowerCase().split(" ");

        handleObj.handler = function( event ) {
            // Don't fire in text-accepting inputs that we didn't directly bind to
            if ( this !== event.target && (/textarea|select/i.test( event.target.nodeName ) ||
            event.target.type === "text") ) {
                return;
            }

            // Keypress represents characters, not special keys
            var special = event.type !== "keypress" && $.hotkeys.specialKeys[ event.which ],
            character = String.fromCharCode( event.which ).toLowerCase(),
            key, modif = "", possible = {};

            // check combinations (alt|ctrl|shift+anything)
            if ( event.altKey && special !== "alt" ) {
                modif += "alt+";
            }

            if ( event.ctrlKey && special !== "ctrl" ) {
                modif += "ctrl+";
            }

            // TODO: Need to make sure this works consistently across platforms
            if ( event.metaKey && !event.ctrlKey && special !== "meta" ) {
                modif += "meta+";
            }

            if ( event.shiftKey && special !== "shift" ) {
                modif += "shift+";
            }

            if ( special ) {
                possible[ modif + special ] = true;

            } else {
                possible[ modif + character ] = true;
                possible[ modif + $.hotkeys.shiftNums[ character ] ] = true;

                // "$" can be triggered as "Shift+4" or "Shift+$" or just "$"
                if ( modif === "shift+" ) {
                    possible[ $.hotkeys.shiftNums[ character ] ] = true;
                }
            }

            for ( var i = 0, l = keys.length; i < l; i++ ) {
                if ( possible[ keys[i] ] ) {
                    return origHandler.apply( this, arguments );
                }
            }
        };
    }

    $.each([ "keydown", "keyup", "keypress" ], function() {
        $.event.special[ this ] = {
            add: keyHandler 
        };
    });
}));
define('internal/sitebuilder/builder/wizzy', [
'jquery',
'internal/sitebuilder/common/log',
'internal/sitebuilder/builder/rangy',
'jquery.hotkeys'
], function($, log) {
    $(function() {
        rangy.init();
        /**
        		 * Surrounds range with a certain tag if possible, returns new element
        		 */
        rangy.rangePrototype.surroundWithTag = function(tag) {
            if (this.canSurroundContents()) {
                var el = document.createElement(tag);
                this.surroundContents(el);
                return el;
            }
            return null;
        };
    });

    var
    /**
    	 * Logs Wizzy
    	 */
    _log = function() {
        if (log && log.trigger) {
            var arr = Array.prototype.slice.call(arguments, 0, arguments.length);

            // if type of log isn't passed in (or matches log type), assume info
            if (typeof log[arr[0]] !== "function") 
                arr.splice(0, 0, "info");

            arr.splice(0, 0, "Wizzy");
            log.trigger.apply(null, arr);
        }
    },

    ELEMENT_NODE = 1,
    TEXT_NODE = 3,
    KEY = {
        LEFT : 37,
        UP : 38,
        RIGHT : 39,
        DOWN : 40,
        RETURN : 13
    },
    CLASSES = {
        BOLD: "wz-bold",
        ITALIC: "wz-italic",
        UNDERLINE: "wz-underline",
        STRIKETHROUGH: "wz-strikethrough"
    },


    utils = {
        /**
        		 * checks to see if char is whitespace, assume &nbsp; (charCode 160) is not whitespace
        		 */
        isWhitespace: function(c) {
            return c && c.charCodeAt(0) !== 160 && c.match(/^\s+|\s+$/);
        },

        /**
        		 * Trim whitespaces from string
        		 */
        trim: function(str) {
            return str.replace(/^\s+|\s+$/g, '');
        },

        /**
        		 * nodeType checker
        		 */
        isTNode: function(node) {
            return node.nodeType === TEXT_NODE;
        },
        isENode: function(node) {
            return node.nodeType === ELEMENT_NODE;
        },

        /**
        		 * Checks whether b is descendant of a, keep traversing up a's ancestors
        		 * Will effectively check which element is more "royal" from an ancestor's perspective
        		 */
        elementContains: function(a, b, escapeEl) {
            if (typeof a.jquery !== "string") 
                a = $(a);

            if (a[0] === escapeEl) 
                return false;
            if (a.find(b)[0]) {
                return true;
            } else {
                utils.elementContains(a.parent(), b, escapeEl);
            }
        },

        /**
        		 * Sibling type is either "nextSibling" or "previousSibling"
        		 * Returns element if found, null otherwise;
        		 */
        getSiblingElement: function(direction, node) {
            if (!node || !utils.isTNode(node)) 
                return null;
            var siblingType = direction === "forward" ? "nextSibling" : "previousSibling";
            while (node[siblingType]) {
                if (!utils.isTNode(node[siblingType])) 
                    return node[siblingType];
                node = node[siblingType];
            }
            return null;
        },

        /**
        		 * Gets the last text node before either an element node or null sibling
        		 */
        getLastTextnode: function(direction, node) {
            var siblingType = direction === "forward" ? "nextSibling" : "previousSibling";
            while (node[siblingType]) {
                if (!utils.isTNode(node[siblingType])) 
                    break;
                node = node[siblingType];
            }
            return node;
        },

        /**
        		 * Selects the node contents, returns endOffset (startOffset should always be zero for els
        		 */
        getNodeEndOffset: function(node) {
            if (typeof node === "undefined") 
                return null;
            var range = rangy.createRange();
            range.selectNodeContents(node);
            return range.endOffset;
        }
    },


    /**
    	 * Allows plugins to register with Wizzy
    	 * In order to utilize this plugin later, the instance of wizzy must call .extend(id, param) - look below
    	 *
    	 * {param} object - id of plugin to register
    	 * {param} object - function at gets called to register plugin
    	 */
    plugins = {},
    register = function(id, fn) {
        _log("Registered Wizzy plugin with ID: " + id);
        plugins[id] = fn;
    },


    /**
    	 * Wizzy creates instance of WizzyRange to handle range. Ideally, there should be only one instance
    	 * of WizzyRange to go with Wizzy since you can set a different range to WizzyRange. We're responsible
    	 * for managing different Rangy ranges and setting it to our WizzyRange instance as necessary.
    	 */
    Wizzy = {
        //==================================++++++++++++++++++++======================================
        /**
        		 * Initializing Wizzy
        		 * {param} {$el} jQuery reference to the content-editable DOM element
        		 */
        init: function($el, data) {
            var self = this;

            // references to content-editable container
            self.$el = $el;
            self.el = $el[0];

            // Holds certain data associated with wizzy (eg. linkInfo)
            self.data = data || {};
            self.data.linksData = self.data.linksData || {};

            self.document = self.el.ownerDocument;
            self.$body = $el.parents('body');

            // Every Wizzy has a WizzyRange, a range traversal adapter
            self.wizzyRange = Object.create(WizzyRange);

            // Indicates that wizzyRange hasn't been set with the current operating range yet
            // This usually means the current selection. At the appropriate time (during exec call),
            self._dirty = true;
            self.bind("rangeChange", function() {
                self._dirty = true;
            });

            // Content-changed delegation
            self.bind("rangeChange", function() {
                var newData, newRange, _applyRange;
                if (!self.currentRange || !self.currentRange.startContainer.parentElement) {
                    self.currentRange = self.getSelectionRange();
                    self.currentData = self.getData();
                    self.getRangeFontSize();
                    return;
                }

                newData = self.getData();
                newRange = self.getSelectionRange();

                if (self.currentData !== newData) {
                    self.trigger("contentChanged", self.currentData);
                    if (self.currentData.length < newData.length) {
                        self.trigger("contentAdded");
                    } else {
                        self.trigger("contentRemoved");
                    }

                    if (self.isCollapsed()) {
                        _applyRange = self.currentRange.cloneRange();
                        if (_applyRange.comparePoint(newRange.endContainer, newRange.endOffset) > 0) {
                            _applyRange.setEnd(newRange.endContainer, newRange.endOffset);
                        } else {
                            _applyRange.setStart(newRange.endContainer, newRange.endOffset);
                        }
                        self.trigger("contentChangedWithRange", _applyRange);
                    }
                }
                self.currentData = newData;
                self.currentRange = newRange;
                self.getRangeFontSize();

            });


            self.utils = utils;
            self.CLASSES = CLASSES;

            self.on = false;

            self.hasHotkeys = $el.constructor.hotkeys;
        },

        getRangeFontSize: function(getDeepestSize) {
            var self = this;
            window.setTimeout(function() {
                var currentRange = self.getSelectionRange(),
                onlyOneFontSize = true,
                fontSize;

                if (currentRange.endContainer !== currentRange.startContainer) {
                    //Determine the font size of the selected text.
                    self.forInRange(function(el) {

                        $el = el.nodeName === '#text' ? $(el.parentNode) : $(el);
                        var style = $el.css("font-size");
                        if (style) {
                            if (!fontSize) {
                                fontSize = parseInt(style, 10);
                            } else {
                                if (fontSize != parseInt(style, 10)) {
                                    onlyOneFontSize = false;
                                }
                            }
                        }
                    }, false);
                }

                //If the selection is contained within a single element, select the node surrounding the cursor to determine the font size.
                if (!fontSize) {
                    var node = currentRange.startContainer.nodeName === '#text' ? $(currentRange.startContainer.parentNode) : $(currentRange.startContainer);
                    // don't get font size from module container if there are child elements
                    // TODO: this is a temp fix for IE
                    if (node.attr('class') && node.attr('class').indexOf('webs-') >= 0 && node.children().length) {
                        node = node.children().first();
                    }
                    fontSize = parseInt(node.css('font-size'), 10);
                }
                //Reset the font size in the tool bar. If the text contains more than 1 font size, set the
                //initial value of the dropdown to ''.
                if (self.data.setFontSize) {
                    self.data.setFontSize(onlyOneFontSize ? fontSize ? fontSize : self.data.defaultFontSize : '');
                }
            }, 1);
        },

        setElement: function(el) {
            if (el.jquery) 
                el = el[0];
            this.el = el;
            this.$el = $(el);
        },

        isDirty: function() {
            return this._dirty;
        },

        /**
        		 * Enables wizzy
        		 */
        show: function() {
            var self = this;
            self.$el.attr('contenteditable', true);
            self.on = true;
            self._bindCore();
            self.trigger('show');
            return self;
        },


        /**
        		 * Disables wizzy
        		 */
        hide: function() {
            var self = this;
            self.$el.attr('contenteditable', false);
            self.on = false;
            self._unbindCore();
            self.trigger('hide');
            self.postProcess();
            return self;
        },


        /**
        		 * Destroys this instance of wizzy, dereference everything for GC
        		 */
        destroy: function() {
            var self = this;
            self.trigger('destroy');
            self.postProcess();
            if (self.$el) {
                self._unbindCore();
                self.$el.data('wizzy', null);
                self.$el = null;
            }
            return true;
        },


        /**
        		 * Returns the HTML data for Wizzy
        		 */
        getData: function() {
            var $clone = this.$el.clone(), $nest, idx = 0;
            $clone.find("ol>span,ul>span").remove();
            while (($nest = $clone.find("ol>ol,ol>ul,ul>ol,ul>ul").eq(0)).length) {
                if ($nest.prev().length === 0) {
                    $("<li><span><br/></span></li>").insertBefore($nest).append($nest);
                } else {
                    $nest.prev().append($nest);
                }
            }
            var self = this, text = utils.trim($clone.html());
            if (text === "<br>") {
                self.$el.html("");
                self.selectNode(self.el);
                text = "";
            }
            return text;
        },
        getLinksData: function() {
            return this.data.linksData;
        },



        //================================== Event Bindings ======================================

        _bindCore: function() {
            var self = this, $el = self.$el;

            $el
            .bind('dblclick.wizzy', function(e) {
                return self._wzDoubleClick(e);
            })
            .bind('click.wizzy', function(e) {
                return self._wzClick(e);
            })
            .bind('mouseup.wizzy', function(e) {
                return self._wzMouseup(e);
            })
            .bind('keyup.wizzy', function(e) {
                return self._wzKeyup(e);
            })
            .bind('keydown.wizzy', function(e) {
                return self._wzKeydown(e);
            });

            if (self.hasHotkeys) {
                $el
                .bind("keydown.wizzy", "tab", function() {
                    return self._hkTab();
                })
                .bind("keydown.wizzy", "shift+tab", function() {
                    return self._hkShiftTab();
                });
            }
        },
        _unbindCore: function() {
            this.$el.unbind('.wizzy');
        },


        _wzDoubleClick: function(e) {
            try {
                this.trigger('rangeChange');
            } catch (err) {
                _log("warn", err);
            }
            return true;
        },
        _wzMouseup: function(e) {
            try {
                this.trigger('rangeChange');
            } catch (err) {
                _log("warn", err);
            }
            return true;
        },
        _wzClick: function(e) {
            try {
                this.trigger('rangeChange');
            } catch (err) {
                _log("warn", err);
            }
            return true;
        },
        _wzKeydown: function(e) {
            var k = e.keyCode;
            if (k === KEY.RETURN) 
                this._wzKeydownReturn();
            return true;
        },
        _wzKeyup: function(e) {
            try {
                this.trigger('rangeChange');
                this.trigger("keyUp", e);
            } catch (err) {
                _log("warn", err);
            }
            return true;
        },


        /**
        		 * Return key capture is used to add a line break if user is in the middle of a line
        		 * and a paragraph if the caret is directly before or after a line break
        		 */
        _wzKeydownReturn: function() {
            /*
            			  Disabling breaks on pressing return key for now.
            			  Will revisit for the cross-browser rich-text editing task
            			 */
            if (0) {
                var s = rangy.getSelection(),
                r = s.getRangeAt(0),
                ancestor = r.commonAncestorContainer,
                prev = ancestor.previousSibling,
                next = ancestor.nextSibling;

                if (r.startOffset === 0 && r.endOffset === 0 && prev && prev.tagName === 'BR') {
                    // Caret is at first position after line break
                    // allow contenteditable to take it's natural course
                    prev.parentNode.removeChild(prev);
                    return true;
                } else if (r.endOffset === ancestor.length && next && next.tagName === 'BR') {
                    // Caret is at last position before line break
                    next.parentNode.removeChild(next);
                    return true;
                } else {
                    var br = document.createElement('br'),
                    endContainer = r.endContainer,
                    n = this.getNode(),
                    newTextNode;


                    // If the caret is at the last position inside a paragraph and the next node is a paragraph
                    // we need to add 2 <br> because the caret doesn't know how to sit in between a <br> and a closing </p>
                    if (r.endOffset === ancestor.length) {
                        if (n.nextElementSibling && n.nextElementSibling.tagName === 'P' && n.tagName === 'P') {
                            this.insertNode(document.createElement('br'));
                        } else {
}
                    }

                    this.insertNode(br);

                    // endContainer's next sibbling is now <br>, br's next sibbling will be new textNode
                    newTextNode = endContainer.nextSibling.nextSibling;
                    r = rangy.createRange();
                    r.setStart(newTextNode, 0);
                    r.setEnd(newTextNode, 0);

                    this.selectRange(r.nativeRange, s);

                    return false;
                }
            }
        },


        _hkTab: function() {
            this.getWizzyRange().init();
            return false;
        },


        _hkShiftTab: function() {
            this.wizzyRange.init();
            return false;
        },





        //================================== Utilities and such ======================================

        /**
        		 * Converts all characters in a string that area above the codepoint 127 into numerical html entities (e.g. &#8220;)
        		 */
        extendedCharsToEntities: function(text) {
            var i, c, cc, newText = "";
            for (i = 0; i < text.length; ++i) {
                cc = text.charCodeAt(i);
                c = cc > 127 ? "&#" + cc + ";" : text.charAt(i);
                newText += c;
            }
            return newText;
        },


        /**
        		 * Generates a "random" string to be used as DOM id
        		 */
        generateId: function() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                var r = Math.random() * 16 | 0,
                v = c === 'x' ? r : r & 3 | 8;
                return v.toString(16);
            }).toUpperCase();
        },


        /**
        		 * IE queryCommandValue returns hex number instead of hex string, needs conversion
        		 */
        queryColorFix: function(value) {
            if ($.browser.msie) {
                value = ((value & 0x0000ff) << 16) | (value & 0x00ff00) | ((value & 0xff0000) >>> 16);
                value = value.toString(16);
                value = "#000000".slice(0, 7 - value.length) + value;
            }
            return value;
        },


        /**
        		 * Grabs the em value if one exist. If not, compute one
        		 */
        getFontSize: function(el) {
            if (!el) {
                _log("warn", "No element defined to get font size");
                return false;
            }

            var size, $el, fz, pfz;
            if (el.style) {
                if (el.style.getPropertyValue) {
                    size = el.style.getPropertyValue('font-size');
                } else if (el.style.fontSize) {
                    size = el.style.fontSize;
                }
            }

            if (size && size.substr( - 2) === 'em') {
                size = parseFloat(size);
            } else {
                $el = $(el);
                fz = parseInt($el.css('font-size'), 10);
                pfz = parseInt($el.parent().css('font-size'), 10);
                size = fz / pfz;
            }
            return size;
        },


        /**
        		 * Sets font size on element with value. If isStep is set, value is an increment
        		 */
        setFontSize: function(el, value, isStep) {
            var self = this,
            applyFontSize = function($el, value, isStep) {
                var size;
                if (isStep) {
                    size = this.getFontSize(el, false);
                    size = parseInt(size * 100 + value * 100, 10) / 100 + 'em';
                } else {
                    size = value + 'px';
                }
                $el.css('font-size', size);
            },
            $el = $(el);

            // Apply the new font size to the element
            applyFontSize($el, value, isStep);

            $el.children().each(function() {
                self.setFontSize($(this), value, isStep);
            });

            return $el;
        },


        /**
        		 * Recursive deletes empty or singular block-level (P and DIV) nodes
        		 */
        clearSurroundingMarkup: function(el) {
            var n, tmp, next = el, prev = el, parent = el;
            var tagsToRemove = ["P", "DIV", "SPAN"];

            do {
                do {
                    tmp = next;
                    next = tmp.next();
                    n = next[0];
                    if (n && ($.inArray(n.tagName, tagsToRemove)) && next.html() === "") {
                        next.remove();
                        next = tmp;
                        continue;
                    }
                }
                while (next[0]);

                do {
                    tmp = prev;
                    prev = tmp.prev();
                    n = prev[0];
                    if (n && $.inArray(n.tagName, tagsToRemove) && next.html() === "") {
                        prev.remove();
                        prev = tmp;
                        continue;
                    }
                }
                while (prev[0]);

                tmp = parent;
                parent = tmp.parent();
                n = parent[0];
                if (n && n === this.el || parent.find(this.el)[0]) 
                    return;
                if (parent.children().length === 1 && $.inArray(n.tagName, tagsToRemove)) {
                    parent.before(tmp);
                    parent.remove();
                    parent = tmp;
                }
            }
            while (parent[0]);
        },




        //================================ SELECTION & RANGES ====================================
        getWizzyRange: function() {
            return this.wizzyRange;
        },
        getRange: function(r) {
            return this.getWizzyRange().getRange();
        },
        getSelectionRange: function(r) {
            var range;
            try {
                range = this.getSelection().getRangeAt(0);
            } catch (e) {
                _log("warn", "Error grabbing rangy range, using root element instead", e);
                range = rangy.createRange();
                range.selectNodeContents(this.el);
            }
            return range;
        },
        getSelection: function() {
            return rangy.getSelection();
        },
        clearSelection: function() {
            this.getSelection().removeAllRanges();
            return this;
        },


        getElement: function() {
            var n = this.getNode();
            if (n && n.nodeType !== ELEMENT_NODE) 
                n = n.parentNode;
            if (this.el !== n && !this.$el.find(n)[0]) 
                return null;
            return n;
        },
        getNode: function() {
            return this.getSelectionRange().commonAncestorContainer;
        },


        insertNode: function(node, replace) {
            var s = this.getSelection(), r = s.getRangeAt(0);
            if (replace) 
                s.deleteFromDocument();
            return r.insertNode(node);
        },


        insertHTML: function(html, replace) {
            var r = this.getSelectionRange(), node = r.createContextualFragment(html);
            var oldR = this.getSelectionRange();
            return r.insertNode(node, replace);
        },


        collapse: function(node, offset) {
            if (typeof node === 'undefined') 
                node = this.el;
            if (typeof offset !== 'number') 
                offset = 0;
            return this.getSelection().collapse(node, offset);
        },
        isCollapsed: function() {
            return this.getSelection().isCollapsed;
        },


        // sets range on wizzyRange
        setRange: function(r) {
            _log("Range was dirty, setting and cleaning range");
            this.getWizzyRange().setRange(r);
            this._dirty = false;
            return this;
        },

        // sets range (saves rave to wizzyRange) only if dirty
        setDirtyRange: function() {
            if (this.isDirty())
                this.setRange();
        },

        // selects current wizzyRange
        select: function(fn) {
            var self = this;
            setTimeout(function() {
                self.selectRange(self.getRange());
                if (typeof fn === "function") 
                    fn.call(self);
            }, 0.01);
            return self;
        },

        selectWord: function() {
            var range, backwards, sel = this.getSelection().nativeSelection;
            if ((window.getSelection) && (sel = window.getSelection()).modify) {

                // Detect if selection is backwards
                sel = window.getSelection();
                range = this.getSelectionRange();
                range.setStart(sel.anchorNode, sel.anchorOffset);
                range.setEnd(sel.focusNode, sel.focusOffset);
                backwards = range.collapsed;
                range.detach();

                // modify() works on the focus of the selection
                var endNode = sel.focusNode, endOffset = sel.focusOffset;
                sel.collapse(sel.anchorNode, sel.anchorOffset);
                if (backwards) {
                    sel.modify("move", "forward", "word");
                    sel.extend(endNode, endOffset);
                    sel.modify("extend", "backward", "word");

                } else {
                    sel.modify("move", "backward", "word");
                    sel.extend(endNode, endOffset);
                    sel.modify("extend", "forward", "word");
                }
            } else if (document.selection) {
                range = document.selection.createRange().duplicate();
                range.moveStart("word", - 1);
                range.moveEnd("word", 1);
                range.select();
            }
        },

        selectRange: function(r) {
            if (r.isValid()) 
                this.getSelection().setSingleRange(r);
        },
        selectAll: function() {
            var range = rangy.createRange();
            range.selectNodeContents(this.el);
            this.selectRange(range);
            return range;
        },
        selectNode: function(node) {
            this.getSelection().selectAllChildren(node);
        },
        saveRange: function() {
            this.savedRange = this.getSelectionRange();
            return this.savedRange;
        },
        restoreRange: function() {
            if (this.savedRange) {
                this.selectRange(this.savedRange);
            }
        },

        //================================== EXTENDING PLUGIN AND EVENTS ======================================

        /**
        		 * Allows the current instance of wizzy to extend functionality
        		 * The plugin callback function is given the instance of wizzy along with necessary parameters
        		 *
        		 * {param} string - ID of the plugin that was registered through Wizzy.register
        		 * {param} object - Object with all parameters for plugin function callback to use
        		 * {return} Boolean - False if there were problems, True if sucessfull. Return is largely dependent on plugin developer
        		 */
        extend: function(id, arg) {
            return plugins[id].call(this, arg);
        },

        /**
        		 * Allows registered plugin to register callbacks that can hook into common callbacks & events in Wizzy
        		 * Piggybacks on jQuery. You can define a binding as follows
        		 *      w.bind("event_name", function(e, params) {  });
        		 */
        bind: function(event, fn) {
            var self = this;
            event = "wizzy-" + event;
            self.$el.bind(event, fn);
            return self;
        },
        trigger: function(event, param) {
            var self = this;
            event = "wizzy-" + event;
            self.$el.trigger(event, param);
            return self;
        }

    },
    /**
    	 * Class which wraps rangy's range. A rangy range instance is associated with WizzyRange.
    	 * Use setRange to update it with a new range
    	 */
    WizzyRange = {

        /**
        		 * Initializting WizzyRange is just setting a range on it
        		 */
        init: function(r) {
            return this.setRange(r);
        },

        /**
        		 * "r" is a rangy range. If none is passed in, we use the selection's range
        		 */
        setRange: function(r) {
            if (r) 
                this.r = r.cloneRange();
            else 
                this.setSelectionRange();
            return this;
        },

        /**
        		 * Sets current range with browser selection
        		 */
        setSelectionRange: function() {
            this.r = rangy.getSelection().getRangeAt(0);
        },

        /**
        		 * Grabs rangy range associated wtih WizzyRange
        		 */
        getRange: function() {
            return this.r;
        },

        /**
        		 * Returns an instance of WizzyRange with a unique copy of the rangy range
        		 */
        clone: function() {
            return Object.create(WizzyRange).init(this.r.cloneRange());
        },

        /**
        		 * Returns a clone of the rangy range
        		 */
        cloneRange: function() {
            return this.getRange().cloneRange();
        },

        /**
        		 * Creates a selection (highlight) with the rangy range instance
        		 */
        select: function() {
            return rangy.getSelection().setSingleRange(this.r);
        },

        /**
        		 * Getters, 's' for anything start (startContainer, startOffset), 'e' for anything end
        		 */
        get: function(which) {
            return this.r[which === "s" ? "startContainer" : "endContainer"];
        },

        /**
        		 * Fires callback on all top-level elements within range
        		 */
        forInRange: function(callback, wrapTextNodes) {
            if (typeof wrapTextNodes === 'undefined') {
                wrapTextNodes = true;
            }
            var self = this;
            var elsArr = this.getElementsInRange(wrapTextNodes);
            $.each(elsArr, function(i, el) {
                callback.call(self, el);
            });
        },

        /**
        		 * Returns array of top level nodes inside range
        		 * If tieEnds is set, dangling textnode endpoints will be wrapped by a span
        		 */
        getElementsInRange: function(tieEnds) {
            var
            elsArr = [],
            span = null,
            r = this.cloneRange(),
            sc = this.get('s'),
            ec = this.get('e'),
            ancestor = r.commonAncestorContainer;

            // Does range fully contain an element?
            if (this._fullyContainsEl()) {
                elsArr.push(this.r.commonAncestorContainer);
                return elsArr;
            }

            // Does range fully contain text node?
            if (this._containsOnlyText()) {
                if (tieEnds) {
                    span = this.getRange().surroundWithTag('span');
                    this.getRange().selectNodeContents(span);
                    this.select();
                    elsArr.push(span);
                }
                return elsArr;
            }

            // Wrap end nodes (if they're text nodes);
            if (tieEnds) 
                this._wrapEndNodes();

            // Finally grab elements
            elsArr = this._getRangeElements(true);

            // Elements could have been created from textnodes in _getRangeElements, needs reselect
            this.select();

            return elsArr;
        },

        /**
        		 * Gets all ancestral elements within range
        		 * If textnode doesn't have ancestral element, wrap with span
        		 */
        _getRangeElements: function(wrapTextNodes) {
            var
            self = this,
            sc = this.get("s"),
            ec = this.get("e"),
            ancestor = this.r.commonAncestorContainer,
            elements = [sc, ec],
            _range, _parent, _siblings;

            this.getRange().getNodes([TEXT_NODE], function(node) {
                // Not interested in empty text nodes
                if (node.data === "") 
                    return;

                while (node) {
                    // We already have this node
                    if ($.inArray(node, elements) !== - 1) 
                        return;

                    _parent = node.parentNode;
                    _siblings = _parent.childNodes;

                    if (_parent === ancestor || self._isNodeInCollection([sc, ec], _siblings)) {
                        if (utils.isTNode(node)) {
                            if (!wrapTextNodes) 
                                return;

                            _range = rangy.createRange();
                            _range.selectNode(node);
                            node = _range.surroundWithTag("span");
                        }
                        elements.push(node);
                        return;
                    }
                    node = node.parentNode;
                }
            });
            return elements;
        },

        /**
        		 * Checks whether a node (or nodes) is in or contained in a nodes collection
        		 */
        _isNodeInCollection: function(node, arr) {
            var i, len;
            if ($.isArray(node)) {
                for (i = 0, len = node.length; i < len; i++) {
                    if (this._isNodeInCollection(node[i], arr)) 
                        return true;
                }
            } else {
                for (i = 0, len = arr.length; i < len; i++) {
                    if (node === arr[i] || $(arr[i]).find(node)[0]) 
                        return true;
                }
            }
            return false;
        },

        /**
        		 * checks to see if a range fully contains an element
        		 * if so, return an array with just that element
        		 */
        _fullyContainsEl: function(el, r) {
            if (!r) 
                r = this.getRange();
            if (!el) 
                el = r.commonAncestorContainer;
            var
            sc = r.startContainer,
            ec = r.endContainer,
            endOff = utils.getNodeEndOffset(ec);

            if (r.endOffset === endOff && r.startOffset === 0 && sc === el && ec === el) {

                // sometimes with a new selection the range only includes text node
                // we need to try to get single-children parent node and use that instead
                if (el.parentNode && el.parentNode.childNodes.length === 1) 
                    el = el.parentNode;
                r.selectNodeContents(el);

                if (!utils.isENode(el)) 
                    r.selectNodeContents(r.surroundWithTag("span"));
                return true;
            }
            return false;
        },

        /**
        		 * Checks to see if the range contains only text nodes and endpoints are sibbling
        		 * if so, *wrap range with span, then return array with just that element
        		 */
        _containsOnlyText: function(r) {
            if (!r) 
                r = this.getRange();
            var sc = r.startContainer, ec = r.endContainer, _node;

            if (utils.isTNode(sc) && utils.isTNode(ec)) {
                if (sc === ec) 
                    return true;

                _node = sc;
                while (_node && utils.isTNode(_node)) {
                    if (_node === ec) {
                        return true;
                    } else {
                        _node = _node.nextSibling;
                    }
                }
            }
            return false;
        },

        /**
        		 * If ends are textnodes, wrap nodes with <span/> up to sibbling's non-textnode
        		 */
        _wrapEndNodes: function() {
            var arr = [];
            var sc = this.get('s'), ec = this.get('e');
            var so = this.r.startOffset, eo = this.r.endOffset;
            var _node, _range, _newStart, _newEnd;

            if (utils.isTNode(sc)) {
                _range = rangy.createRange();
                _range.setStart(sc, so);
                /* jshint ignore:start */
                if (_node = utils.getSiblingElement("forward", sc)) {
                    _range.setEndBefore(_node);
                } else {
                    _node = utils.getLastTextnode("forward", sc);
                    _range.setEnd(_node, _node.length);
                }
                /* jshint ignore:end */
                _newStart = _range.surroundWithTag('span');
                arr.push(_newStart);
            }

            if (utils.isTNode(ec)) {
                _range = rangy.createRange();
                _range.setEnd(ec, eo);
                /* jshint ignore:start */
                if (_node = utils.getSiblingElement("backward", ec)) {
                    _range.setStartAfter(_node);
                } else {
                    _node = utils.getLastTextnode("backward", ec);
                    _range.setStart(_node, 0);
                }
                /* jshint ignore:end */
                _newEnd = _range.surroundWithTag('span');
                arr.push(_newEnd);
            }

            if (!_newStart) 
                _newStart = sc;
            if (!_newEnd) 
                _newEnd = ec;

            _range = rangy.createRange();
            _range.selectNodeContents(_newEnd);
            _range.setStart(_newStart, 0);
            this.r = _range;
            this.select();

            return arr;
        }

    };


    //========================= PLUGIN FOR RANGES AND EXEC[COMMAND] CALLS ======================
    register("RangeExec", function() {
        // Extending Wizzy Methods
        $.extend(this, {
            _styleStringToMap: function(str) {
                var result = {}, entry, attr;
                if (str && str.split) {
                    entry = str.split(';');
                    for (var i = 0, len = entry.length; i < len; i++) {
                        attr = entry[i].split(':');
                        if (attr[0] && attr[1]) 
                            result[attr[0]] = attr[1];
                    }
                }
                return result;
            },
            _styleMapToString: function(map) {
                var str = "";
                for (var attr in map) {
                    if (map.hasOwnProperty(attr)) 
                        str += attr + ":" + map[attr] + "; ";
                }
                return str;
            },
            _extendElementStyles: function(el1, el2) {
                if (!el1.jquery) 
                    el1 = $(el1);
                if (!el2.jquery) 
                    el2 = $(el2);

                return $.extend(this._styleStringToMap(el1.attr("style")), this._styleStringToMap(el2.attr("style")));
            },
            _fixNestedSpans: function(parentNode) {
                if (typeof parentNode === "undefined") 
                    parentNode = this.el;
                var frag, arr = [], span, child, spans = parentNode.getElementsByTagName('span');
                var $child, $parent;

                // Some older IEs may not like createDocumentFragment
                if (!document.createDocumentFragment) 
                    return;

                // Stabilise spans collection in array
                for (var i = 0, iLen = spans.length; i < iLen; i++) 
                    arr[i] = spans[i];

                for (i = 0; i < iLen; i++) {
                    span = arr[i];

                    // if element has only one child and that child is a span, we can replace the node with child node
                    child = span.firstChild;
                    $parent = $(span);
                    if (span.childNodes.length === 0) {
                        $parent.remove();
                        continue;
                    } else if (span.children.length === 1 && child.nodeName === "SPAN") {


                        // To be safe, we'll only replace nodes that either doesn't have any attribute
                        if (span.attributes.length > 0) 
                            continue;

                        $child = $(child);

                        // We just need to extend current node style with child node style
                        if (span.getAttribute("style") !== null) {
                            // if current node has a font size, we override it with absolute (px) size
                            if (span.style.fontSize.length) 
                                child.style.fontSize = $child.css("font-size");

                            // extend parent style with child style, throw style on child
                            $child.attr("style", this._styleMapToString(this._extendElementStyles($parent, $child)));
                        }

                        frag = document.createDocumentFragment();
                        frag.appendChild(span.firstChild);

                        // Replace span with its content in the fragment
                        span.parentNode.replaceChild(frag, span);
                    }
                }
                return parentNode.innerHTML;
            },
            preProcess: function() {
                this._fixNestedSpans();
                this.$el.find("span").each(function(i, el) {
                    var style;
                    el = $(el);

                    style = el.css("font-weight");
                    if (typeof style !== "string") 
                        style = style.toString();
                    if (style.match(/bold|700/)) {
                        el.css("font-weight", "").addClass(CLASSES.BOLD);
                    }

                    style = el.css("font-style");
                    if (style.match(/italic/)) 
                        el.css("font-style", "").addClass(CLASSES.ITALIC);

                    style = el.css("text-decoration");
                    if (style.match(/underline/)) 
                        el.addClass(CLASSES.UNDERLINE);
                    if (style.match(/line-through/)) 
                        el.addClass(CLASSES.STRIKETHROUGH);
                    el.css("text-decoration", "");
                    if (el.attr("style") === "") 
                        el.removeAttr("style");
                });

                this._cssNormalizer = this._createApplier("NULL");
            },

            /**
            			 * Use rangy's cssClassApplier's postApply method to normalize nodes
            			 */
            postProcess: function() {
                if (this.$el.parents('body').length === 0) 
                    return;
                var r = rangy.createRange();
                r.selectNodeContents(this.el);
                this._cssNormalizer.postApply(r.getNodes([TEXT_NODE]), r, false);
                this._fixNestedSpans();
            },

            /**
            			 * This method gets called externally and acts as a preprocessor for many
            			 * [contenteditable] built-in functionality.
            			 */
            exec: function(cmd, value) {
                _log("Execute Command:  " + cmd + "  " + value);

                switch (cmd) {
                case 'IncreaseFontSize':
                case 'wzIncreaseFontSize':
                    this.wzChangeFontSize('grow');
                    break;
                case 'DecreaseFontSize':
                case 'wzDecreaseFontSize':
                    this.wzChangeFontSize('shrink');
                    break;
                case 'wzSetFontSize':
                    this.wzChangeFontSize(value);
                    break;
                case 'ForeColor':
                case 'wzForeColor':
                    this.wzForeColor(value);
                    break;
                case 'Link':
                case 'wzLink':
                    this.wzCreateLink(value);
                    break;
                case 'Unlink':
                case 'wzUnlink':
                    this.wzRemoveLink();
                    break;
                case 'List':
                case 'wzList':
                    this.wzList(value);
                    break;
                case 'Bold':
                case 'bold':
                    this.wzBold();
                    break;
                case 'Italic':
                case 'italic':
                    this.wzItalic();
                    break;
                case 'Underline':
                case 'underline':
                    this.wzUnderline();
                    break;
                case 'Strikethrough':
                case 'strikethrough':
                    this.wzStrikethrough();
                    break;
                case 'Indent':
                case 'Outdent':
                case 'indent':
                case 'outdent':
                    this.wzBlockquote(cmd);
                    break;
                case 'RemoveFormat':
                    this.wzRemoveFormat();
                    break;

                default:
                    document.execCommand(cmd, false, value);
                }
            },


            forInRange: function(fn, wrapTextNodes) {
                var self = this;
                var el = this.el;
                var wizzyRange = this.getWizzyRange();

                if (typeof wrapTextNodes === 'undefined') {
                    wrapTextNodes = true;
                }

                // In case wizzyRange doesn't have the most updated range, set (and normalize)
                this.setDirtyRange();

                // select range just in case its not highlighted
                if (this.isCollapsed()) 
                    this.select();

                if (this.getRange().collapsed) 
                    return this;

                wizzyRange.forInRange(function(_el) {
                    if ($.contains(_el, el)) {
                        // For some reason, wizzy is setting the color on a parent of the
                        // content-editable div, which results in the color not being persisted because
                        // it isn't part of the extracted html.
                        var children = self.$el.children();
                        _el = children.length === 1 ? children[0] : el;
                    }
                    // Filter for the case when element to process is instatiated element
                    if (_el === el && wrapTextNodes) {
                        // Can't find an element, so just surround contents of el with a span
                        self.getRange().selectNodeContents(el);
                        _el = self.getRange().surroundWithTag('span');
                    }
                    fn.call(self, _el);
                }, wrapTextNodes);
                self.select();

                return this;
            },


            getListType: function(style) {
                return (style === 'circle' || style === 'disc' || style === 'square') ? 'Unordered' : 'Ordered';
            },

            wzList: function(style) {
                this.setDirtyRange();
                this.select(function() {
                    var
                    currentStyle,
                    range = this.getRange(),
                    startContainer = range.startContainer,
                    $start = $(startContainer),
                    parentLists = $start.parents('ol,ul'),
                    parentList = parentLists.first(),
                    listType = this.getListType(style);

                    // it's possible that the user has the whole container selected
                    // we need to look at the only child to see if there's a list
                    if (parentList.length === 0 && startContainer === this.el) {
                        var childList = $start.children("ul, ol");
                        if (childList.length === 1) 
                            parentList = childList;
                    }

                    if (style === 'none') {

                        // Applying the same list style will remove the list
                        if (parentList.length === 1) {
                            listType = this.getListType(parentList.css("list-style-type"));
                            this.exec('Insert' + listType + 'List');
                        }
                        this.setRange();
                    } else if (parentList.length > 0) {
                        if (((listType === 'Ordered' && parentList[0].tagName === 'UL') ||
                        (listType === 'Unordered' && parentList[0].tagName === 'OL'))) {
                            // They were in a list of a different type. Change the type and style.
                            this.exec('Insert' + listType + 'List');
                        }
                        currentStyle = parentList.eq(0).css('list-style-type');
                        parentList.eq(0).removeClass('style-' + currentStyle).addClass('style-' + style);
                        this.setRange();
                    } else {
                        // They were not in a list
                        this.exec('Insert' + listType + 'List');
                        var currNode = this.getNode();
                        if (currNode !== this.el) {
                            parentList = $(currNode).parents('ol,ul').first();
                        } else {
                            parentList = $(currNode).children('ol,ul').eq(0);
                        }
                        currentStyle = parentList.css('list-style-type');
                        parentList.removeClass('style-' + currentStyle).addClass('style-' + style);
                        this.setRange();
                        this.clearSurroundingMarkup(parentList);
                    }
                });
            },


            wzChangeFontSize: function(val) {
                if (typeof val === 'undefined') 
                    return false;
                if (typeof val === 'string') {
                    if (val === 'grow') 
                        val = 0.1;
                    else if (val === 'shrink') 
                        val = - 0.1;
                }

                this.forInRange(function(el) {
                    this.setFontSize(el, val, val < 1.0);
                });

                return false;
            },


            wzForeColor: function(hex) {
                this.forInRange(function(el) {

                    // Recursively ensure this property is set for all children.
                    // Without this, IE doesn't seem to cope.
                    var setColorRecursive = function(el, color) {
                        $el = $(el);
                        $el.css("color", color);
                        $el.children().each(function() {
                            setColorRecursive(this, color);
                        });
                    };

                    setColorRecursive(el, hex);
                });
            },

            wzRemoveFormat: function() {
                var wizzyFontSize = parseInt(this.$el.css("font-size"), 10);
                var wizzyColor = this.$el.css("color");
                // Removes all style and reset font size and color
                this.forInRange(function(el) {
                    if (!utils.isENode(el)) 
                        return;

                    var
                    $el = $(el).removeAttr("style"),
                    fontSize = parseInt($el.css("font-size"), 10),
                    color = $el.css("color");
                    $el.find("span").removeAttr("style");

                    if (wizzyFontSize !== fontSize) 
                        this.setFontSize($el[0], wizzyFontSize / fontSize);
                    if (color !== wizzyColor) 
                        $el.css("color", wizzyColor);
                });

                // Removes all wizzy classes
                if (this.appliers) {
                    for (var klass in this.appliers) {
                        if (this.appliers.hasOwnProperty(klass)) {
                            this.appliers[klass].undoToRange(this.getRange());
                        }
                    }
                }
            },


            /**
            			 * Creates and executes class applier
            			 */
            _createApplier: function(klass) {
                return rangy.createCssClassApplier(klass, {
                    normalize: true,
                    applyToEditableOnly: true
                });
            },
            _exec: function(klass) {
                try {
                    var self = this, nullChar;
                    var selRange = self.getSelectionRange();
                    var range = selRange.cloneRange();
                    var collapsed = self.isCollapsed();

                    var applier = self.appliers = self.appliers || {};
                    applier[klass] = applier[klass] || self._createApplier(klass);

                    // creates range bookmark for IE
                    self.$el.focus();
                    var ieRange = document.selection && document.selection.createRange().duplicate();
                    var bookmark;

                    // If selection is collapsed, we want to apply bold to the next character the user inserts
                    // This is done by creating a null-character textnode, selecting it, and calling class applier
                    // However, in IE, we can't select a null character, therefore the class applier will be
                    // performed on the current word (this is how gmail does it)
                    if (collapsed) {
                        if (!$.browser.msie) {
                            nullChar = document.createTextNode(String.fromCharCode(0));
                            if ($.browser.mozilla) 
                                nullChar = document.createTextNode(" ");
                            self.insertNode(nullChar);
                            range.selectNodeContents(nullChar);
                        } else {
                            bookmark = ieRange.getBookmark();
                            self.selectWord();
                            range = self.getSelectionRange();
                        }
                        self.selectRange(range);
                    }
                    applier[klass].toggleSelection();
                    self._dirty = true;
                    self.$el.focus();

                    // IE loses selection information after the toggle, needs to reselect
                    if ($.browser.msie) {
                        setTimeout(function() {
                            try {
                                self.$el.focus();
                                ieRange.moveToBookmark(bookmark);
                                ieRange.select();
                                self.trigger("rangeChange");
                            } catch (err) {}
                        }, 1);
                    } else if ($.browser.mozilla && collapsed) {
                        range = self.getSelectionRange();
                        range.deleteContents();
                    }
                } catch (err) {}
            },
            wzBold: function() {
                this._exec(CLASSES.BOLD);
            },
            wzItalic: function() {
                this._exec(CLASSES.ITALIC);
            },
            wzUnderline: function() {
                this._exec(CLASSES.UNDERLINE);
            },
            wzStrikethrough: function() {
                this._exec(CLASSES.STRIKETHROUGH);
            },

            wzClick: function() {
                this._wzClick(null);
            },



            wzCreateLink: function(linkObj) {
                this.setDirtyRange();
                this.select(function() {
                    var linkId, node = this.getElement(), isCollapsed = this.isCollapsed();

                    if (isCollapsed) {
                        if (!node || node.tagName !== 'A') 
                            return false;
                        this.selectNode(node);
                    }
                    var href = typeof linkObj.url === "string" && linkObj.url ? linkObj.url : "#";
                    this.exec('CreateLink', href);

                    // Firefox doesn't wrap link inside range properly after CreateLink
                    node = this.getElement();
                    if (node.nodeName !== "A") {
                        var r = this.getSelectionRange();
                        if (r.endContainer.nodeName === "A") {
                            node = r.endContainer;
                        } else if (r.endContainer.parentNode.nodeName === "A") {
                            node = r.endContainer.parentNode;
                        } else {
                            node = $(node).find('a');
                            if (node.length === 0) {
                                return false;
                            }
                            node = node.get(0);
                        }
                    }

                    // stores link information with its id in the markup as the key
                    // linkID is either passed in, preexisting, or generated (in that order)
                    linkId = linkObj.id || (node && node.tagName === 'A' && node.id) || this.generateId();
                    node.id = linkId;
                    this.data.linksData[linkId] = linkObj;

                    this.selectNode(node);
                    this.trigger("linkCreated", {
                        node: node,
                        data: linkObj
                    });

                    try {
                        this.trigger('linkChange', linkObj);
                        this.trigger('rangeChange');
                    } catch (err) {
                        _log("warn", err);
                    }
                });
            },


            wzRemoveLink: function() {
                var node = this.getElement();

                try {
                    if (node.tagName === 'A') {
                        var nodeId = node.id;
                        if (nodeId && this.data.linksData[nodeId]) {
                            delete this.data.linksData[nodeId];
                        }

                        this.selectNode(node);
                        this.exec('unlink');
                        this.trigger('linkChange');
                        this.trigger('rangeChange');
                    }
                } catch (err) {
                    _log("warn", err);
                }
            },


            wzBlockquote: function(which) {
                var $n = $(this.getNode()), $list = $n.parents('ol,ul').first(), $block;

                if ($list.length) {
                    // if we're in a list, let contenteditable handle list creation
                    document.execCommand(which, true, true);
                } else {
                    var selection = rangy.saveSelection();
                    if (which === "indent") {
                        $block = $n.parents('p,div').first();
                        if ($block[0] === this.el) {
                            $block.html("<p>" + $block.html() + "</p>");
                            $block = $block.children('p');
                        }
                        $block.wrap("<blockquote/>");
                    } else if (which === "outdent") {
                        $block = $n.parents('blockquote').first();
                        $block.before($block.html());
                        $block.remove();
                    }
                    rangy.restoreSelection(selection);
                    rangy.removeMarkers(selection);
                }
            }
        });

        this.preProcess();
    });






    /* Object.create is native, but not in older browsers */
    if (typeof Object.create !== "function") {
        Object.create = (function () {
            var F = function() {};
            return function (o) {
                F.prototype = o;
                return new F();
            };
        })();
    }

    // The jQuery plugin
    $.fn.wizzy = function(method) {
        return this.each(function() {
            var $this = $(this),
            w = $this.data('wizzy');

            if (w) {
                _log("warn", "Wizzy called on an element that already has wizzy!");
            } else {
                w = Object.create(Wizzy);
                w.init($this, method);
                w.extend("RangeExec"); // Add the RangeExec plugin

                $this.data('wizzy', w);
            }
        });
    };


    return {
        register: register,
        plugins: plugins,
        utils: utils
    };
});

define('internal/sitebuilder/builder/wizzy.ui', [
'jquery',
'internal/sitebuilder/builder/wizzy'
], function($, Wizzy) {

    // Link tooltip
    var $linkHoverText = $('<a/>').addClass('w-wizzy-anchor-text').html("http://"),
    $linkHoverChange = $('<a/>').addClass('w-wizzy-anchor-change').html("Change").attr("title", "Change Link"),
    $linkHoverRemove = $('<a/>').addClass('w-wizzy-anchor-remove').html("Remove").attr("title", "Remove Link"),
    $linkHover = $('<div/>').addClass('w-wizzy-anchor').append(
    "Go to link: ", $linkHoverText, ' - ',
    $linkHoverChange, ' - ', $linkHoverRemove
    );


    /**
     * Plugin to hook UI into Wizzy editor
     * {param} object - instance of wizzy
     * {param} object {
     *    toolbar:    required toolbar instance
     * }
     */
    Wizzy.register('ui', function(arg) {
        var
        w = this,
        tb = [],
        toolbar = arg.toolbar,
        $toolbar = arg.ui ? arg.ui : arg.toolbar.get$Toolbar(),
        // Flags
        enableTypefaceUpdate = arg.enableTypefaceUpdate || false,
        enableAlignment = typeof arg.enableAlignment !== "undefined" ? arg.enableAlignment : true,
        iDropper = toolbar.find('.w-tbui-wzForeColor').data('iDropper'),
        colorState = toolbar.find('.w-icon-text_color .w-colorstate'),
        linkGroup = toolbar.find('.w-wizzy-linkgroup'),
        linkButton = linkGroup.find('.w-wizzy-linkbutton'),
        unlinkButton = linkGroup.find('.w-icon-unlink'),
        fontsDropdown = toolbar.getElement('Fonts Dropdown'),
        fontsController = fontsDropdown.data('dropdown'),
        chromeController = parent.require('internal/sitebuilder/builderChrome/chromeController'),
        $canvasDoc = parent.bldr.pageController.dom.$doc,
        $chromeDoc,
        insideIframe = true,
        range;

        if (chromeController.dom) 
            $chromeDoc = chromeController.dom.doc;
        else 
            $chromeDoc = $(document);

        /**
        	 * For exec commands triggered by the UI that requires more than one click
        	 * Color picker, for example, requires a click to open menu and another to select color.
        	 * The first click will clear the selection in IE. We need to manually setRange when
        	 * we lose focus to the iframe as to preserve our most recent selection
        	 */
        $chromeDoc.bind('mousedown.wizzyui', function() {
            w.setDirtyRange();
            w.select();
        });

        // Handles link creation
        // cloned from handlebars.helpers.js
        var renderLinkAttributes = function(linkInfo) {
            var href, classes = [], attributes = '';

            if (linkInfo.lightbox) {
                if (linkInfo.enabled)
                    href = "#";
            } else if (linkInfo.url) {
                href = linkInfo.url;
            } else if (linkInfo.email) {
                href = "mailto:" + linkInfo.email;
            }

            if (href && typeof Handlebars !== "undefined") 
                href = Handlebars.Utils.escapeExpression(href);
            if (linkInfo.newWindow) 
                classes.push("w-link-new-window");

            if (href) 
                attributes += ' href="' + href + '"';

            return {
                href: href,
                classes: classes
            };
        };
        var linkCreated = function(e, linkObj) {
            var linkInfo = linkObj.data;
            var linkAttrs = renderLinkAttributes(linkInfo);
            var $node = $(linkObj.node);
            $node.attr("href", linkAttrs.href).addClass(linkAttrs.classes.join(" "));
            $node.data("disableLink", true);
        };
        this.$el.find(".w-link-new-window").data("disableLink", true);

        // Link and Unlink buttons and tooltip
        $linkHover.appendTo(w.$body).removeClass("active").unbind('mousedown').mousedown(function() {
            return false;
        });
        var updateLinkIndicator = function(node) {
            if (linkGroup.length) {
                var isOverAnchor = node && node.tagName && node.tagName.toLowerCase() === "a";
                if (!isOverAnchor) 
                    isOverAnchor = node = $(node).parents('a')[0];
                if (isOverAnchor) {
                    toolbar.show({
                        "wzLink": w.data.linksData[node.id]
                    });
                    linkGroup.removeClass('canceled');

                    var $node = $(node),
                    nodeOffset = $node.offset(),
                    anchorUrl = node.href.length < 45 ? node.href : node.href.substr(0, 43) + '...',
                    left = nodeOffset.left - 10, top = nodeOffset.top + $node.height(),
                    linkHoverWidth, LinkHoverOffset, boundaryWidth;

                    $linkHoverText
                    .html(anchorUrl)
                    .attr('title', node.href)
                    .unbind('mousedown').mousedown(function() {
                        window.open(anchorUrl, node.id || '_new');
                    });

                    // Place linkHover in view to calculate correct width and viewport boundaries
                    $linkHover.addClass('active'); // This causes linkHover to flicker on cursor changes .css({ left: 0, top: 0});
                    (function() {
                        linkHoverWidth = $linkHover.width();
                        linkHoverOffset = nodeOffset.left;
                        boundaryWidth = parent.require('internal/sitebuilder/builderChrome/chromeController').pageBoundary.width;
                        if (boundaryWidth && linkHoverWidth + linkHoverOffset > boundaryWidth) {
                            left = boundaryWidth - linkHoverWidth - $node.width();
                        }
                        $linkHover.css({
                            left: left,
                            top: top
                        });
                    }).deferFn();
                } else {
                    toolbar.show({
                        "wzLink": {}
                    });
                    linkGroup.addClass('canceled');

                    $linkHover.removeClass('active');
                }
                linkButton.removeClass('active');
            }
        };

        // setting the correct typeface in the dropdown
        var updateDropdownTypeface = function(node) {
            if (enableTypefaceUpdate && fontsDropdown && fontsController) {
                if (node && node.tagName.toLowerCase() === "font") {
                    var face = node.getAttribute("face");

                    if (face) {
                        fontsController.setOption(face.replace(/'/g, ''));
                    } else {
                        updateDropdownTypeface(node.parentElement);
                    }
                } else {
                    fontsController.setOption('inherit');
                }
            }
        };

        // Filters out icons that have bidirectional relationship with content-editable
        $toolbar.find('a.w-btn-icon').each(function() {
            var $self = $(this),
            prefix = 'w-icon-';

            if ($self.hasClass(prefix + 'bold')) {
                tb['bold'] = $self;
            } else if ($self.hasClass(prefix + 'italic')) {
                tb['italic'] = $self;
            } else if ($self.hasClass(prefix + 'underline')) {
                tb['underline'] = $self;
            } else if ($self.hasClass(prefix + 'strike')) {
                tb['Strikethrough'] = $self;
            }

            if (enableAlignment) {
                if ($self.hasClass(prefix + 'align_left')) {
                    tb['JustifyLeft'] = $self;
                } else if ($self.hasClass(prefix + 'align_center')) {
                    tb['JustifyCenter'] = $self;
                } else if ($self.hasClass(prefix + 'align_right')) {
                    tb['JustifyRight'] = $self;
                } else if ($self.hasClass(prefix + 'align_justify')) {
                    tb['JustifyFull'] = $self;
                }
            }
        });

        var rangeChange = function() {
            $linkHoverChange.unbind('mousedown').mousedown( function() {
                linkButton.click();
            });
            $linkHoverRemove.unbind('mousedown').mousedown( function() {
                unlinkButton.click();
            });

            var node = w.getElement();

            // Font Color
            var color = w.queryColorFix(document.queryCommandValue("ForeColor"));
            iDropper.set(color, true);
            colorState.css('background-color', color);

            updateLinkIndicator(node);

            updateDropdownTypeface(node);

            // Updates UI with proper statead
            var states = ['bold', 'italic', 'underline', 'Strikethrough', 'JustifyLeft', 'JustifyCenter', 'JustifyRight', 'JustifyFull'];
            for (var id in states) {
                if (tb[states[id]]) {
                    if (document.queryCommandState(states[id])) {
                        tb[states[id]].addClass('active');
                    } else {
                        tb[states[id]].removeClass('active');
                    }
                }
            }
        };

        var trackCursor = $('<span style="display:inline !important;"></span>');
        var rangeFocus = function(e) {
            var $win = parent.bldr.pageController.dom.$win;
            var cc = parent.require('internal/sitebuilder/builderChrome/chromeController');
            var range;
            try {
                range = w.getSelectionRange();
                if (range && cc) {
                    range.collapse(false);
                    range.insertNode(trackCursor[0]);

                    var caretTop = cc.iframeAbsY(trackCursor.offset().top);
                    var iframeHeight = cc.pageBoundary.bottom;
                    var caretMaxY = iframeHeight - 100;
                    trackCursor.remove();

                    if (caretTop > caretMaxY) {
                        $win.scrollTop($win.scrollTop() + (caretTop - caretMaxY));
                    }
                }
            } catch (err) {}
        };

        w.bind('show', rangeChange);
        w.bind('rangeChange', rangeChange);
        w.bind('linkCreated', linkCreated);
        w.bind('keyUp', rangeFocus);
        w.bind('hide', function() {
            $linkHover.removeClass('active');
        });
        w.bind('destroy', function() {
            $linkHover.removeClass('active');
            $canvasDoc.unbind('mousedown.wizzyui');
            $chromeDoc.unbind('mousedown.wizzyui');
        });

        return true;
    });

});
/* jshint ignore:start */
define('internal/sitebuilder/builder/htmlParser', [
'spine',
'internal/sitebuilder/common/log'
], function(Spine, log) {

    var HtmlParser = Spine.Class.create({
        init: function(source, options) {
            this.source = this.originalSource = source;
            this.options = options;
            this.queue = [];
        },

        readTree: function() {
            var token = this.next();
            if (!token) 
                return null;
            if (token.endTag) {
                this.requeue(token);
                return null;
            }
            var continueParsing;
            if (!(token.type === "text" || token.type === "comment" || token.selfClosing)) {
                token.children = [];
                var child;
                var closing;
                continueParsing = true;
                while (continueParsing) {
                    while (child = this.readTree()) {
                        token.children.push(child);
                    }
                    closing = this.next();
                    if (!closing || !(closing.endTag) || !(closing.tagName === token.tagName)) {
                        if (closing) {
                            if (!this.canIgnoreClosingTag(token, closing)) {
                                this.requeue(closing);
                                continueParsing = false;
                            }
                        } else {
                            continueParsing = false;
                        }
                    } else {
                        continueParsing = false;
                    }
                }
            }
            if (this.options.supportLists && token.type === "comment" && token.content === "<![if !supportLists]>") {
                // Ignore everything until <![endif]>
                // Perhaps we should be keeping account of the nesting of <![if ???]><![endif]> tags?
                while (token = this.next()) {
                    if (token.type === "comment" && token.content === "<![endif]>")
                        return this.readTree();
                }
            }
            return token;
        },

        requeue: function(token) {
            this.queue.push(token);
        },

        parseStyleRules: function(text) {
            var matches, styles = {};
            while (text && (matches = text.match(/^\s*([^:\s]*)\s*:\s*([^;]*)\s*;?\s*([\s\S]*)$/))) {
                text = matches[3];
                styles[matches[1]] = matches[2];
            }
            return styles;
        },

        next: function() {
            if (this.queue.length > 0) {
                token = this.queue.shift();
                return token;
            }
            if (this.source == "") 
                return null;
            var token = null, matches;
            if (matches = this.source.match(this.patterns.tag)) {
                var tag = matches[1];
                this.source = matches[2];
                matches = tag.match(/^<(\/?)((?:(?:\w|\-)+:)?\w+)([\s\S]*)\s*(\/?)\s*>$/);
                var endTag = matches[1] === "/";
                var selfClosing = matches[4] === "/";
                var tagName = matches[2].toLowerCase();
                var tagAttrs = matches[3];
                var attrs = {};
                while (matches = tagAttrs.match(/^\s*((?:(?:\w|\-)*:)?(?:\w|\-)+)(?:\s*=\s*((?:(?:\"[^\"]*\")|(?:\'[^\']*\')|(?:[^\s\/\>]*))))?([\s\S]*)$/)) {
                    var value = matches[2] || "";
                    value = value.replace(/\"(.*)\"/, "$1").replace(/'(.*)\'/, "$1");
                    attrs[matches[1]] = value;
                    tagAttrs = matches[3];
                }
                if ("style" in attrs) {
                    var styleSource = attrs.style;
                    if (matches = styleSource.match(/^\"([\s\S]*)\"$/)) 
                        styleSource = matches[1];
                    if (matches = styleSource.match(/^\'([\s\S]*)\'$/)) 
                        styleSource = matches[1];
                    attrs.style = this.parseStyleRules(styleSource);
                }
                if (this.selfClosingTags.indexOf(tagName) != - 1) 
                    selfClosing = true;
                token = {
                    type: "tag",
                    tagName: tagName,
                    selfClosing: selfClosing,
                    endTag: endTag,
                    attrs: attrs,
                    originalSource: tag
                };
            } else if (matches = this.source.match(this.patterns.comment)) {
                this.source = matches[2];
                token = {
                    type: "comment",
                    content: matches[1],
                    originalSource: matches[1]
                };
            } else if (matches = this.source.match(this.patterns.text)) {
                this.source = matches[2];
                token = {
                    type: "text",
                    content: matches[1],
                    originalSource: matches[1]
                };
            } else if (matches = this.source.match(this.patterns.xml)) {
                this.source = matches[2];
                return this.next();
            } else {
                log.trigger("htmlParser", "error", "Error parsing: " + this.source);
                this.source = "";
            }
            return token;
        },

        /*
        	 * Using REGEX for HTML parsing, even though someone felt the need to write the following:
        	 * http://stackoverflow.com/questions/1732348/regex-match-open-tags-except-xhtml-self-contained-tags/1732454#1732454
        	 */
        patterns: {
            xml: /^(<\?xml[^>]*>)([\s\S]*)$/,
            tag: /^(<\/?(?:(?:\w|\-)*:)?\w+(?:\s+(?:(?:\w|\-)+:)?(?:\w|\-)+(?:\s*=\s*(?:(?:\"[^\"]*\")|(?:\'[^\']*\')|(?:[^\s\/\>]*))))*\s*(?:\/?)\s*>)([\s\S]*)$/,
            comment: /^(<!(?:(?:--(?:[^-]|(?:-[^-]))*--)|(?:[^>-]|(?:-[^->]))|(?:\[[\w]+]--))*>)([\s\S]*)$/,
            text: /^([^<]+)([\s\S]*)$/
        },

        selfClosingTags: ["img", "br", "wbr", "hr", "meta", "link", "input", "area", "base", "basefont", "col", "frame", "param"],

        canIgnoreClosingTag: function(parent, child) {
            if (child.tagName === "font") 
                return true;
            return false;
        }

    });

    return HtmlParser;

});
/* jshint ignore:end */
;
/* jshint ignore:start */
define('internal/sitebuilder/builder/wizzy.paste', [
'jquery',
'spine',
'internal/sitebuilder/builder/htmlParser',
'internal/sitebuilder/builder/wizzy'
], function($, Spine, htmlParser, Wizzy) {

    var PasteFromWord = Spine.Class.create({
        init: function(source, linksData) {
            this.listRules = {};
            this.linksData = linksData;
            this.originalSource = source;
            this.tokenizer = htmlParser.init(source, {
                supportLists: true
            });
        },

        pasteToWizzy: function(wizzy) {
            var range = wizzy.getRange(),
            self = this;
            var pasteDeferred = $.Deferred();
            this.process(function(generatedHtml) {
                if (generatedHtml === "") {
                    pasteDeferred.resolve();
                    return;
                }
                range.deleteContents();
                wizzy.insertHTML(generatedHtml + "<span id='pasteKaret'></span>", true);
                range.selectNode($("#pasteKaret").removeAttr("id")[0]);
                wizzy.selectRange(range);
                pasteDeferred.resolve();
            });
            return pasteDeferred.promise();
        },

        process: function(callback) {
            this.list = {
                l: 0,
                level: 0,
                closers: []
            };
            var tree,
            text = "",
            keepGoing = true,
            token;
            while (keepGoing) {
                while (keepGoing && (tree = this.tokenizer.readTree())) {
                    text += this.treeToHtml(tree, "");
                    if (tree.type == "tag" && tree.tagName == "html") {
                        keepGoing = false;
                    }
                }
                if (keepGoing) {
                    token = this.tokenizer.next();
                    while (token && token.endTag)
                        token = this.tokenizer.next();
                    if (token)
                        this.tokenizer.requeue(token);
                    else
                        keepGoing = false;
                }
            }
            if (typeof callback === "function")
                callback(text);
            return text;
        },

        treeToHtml: function(tree, parentClasses) {
            var self = this;
            if (tree.type == "text") {
                if (tree.content.match(/^\s*$/)) 
                    return " ";
                return tree.content;
            }
            if (tree.type == "comment") 
                return "";
            var buffer = "";
            var clazz = "";
            var fontSize, fontColor;
            var childHTML = "";
            var tagHTML = "";
            switch (tree.tagName) {
            case "style":
                this.processStyleTag(tree);
            case "meta":
            case "link":
            case "o:p":
                return "";
            case "br":
                return "<br/>";
            case "table":
                // TODO: do this
                return "[[PASTING TABLES IS NOT SUPPORTED]]";
            case "b":
            case "i":
            case "u":
            case "s":
                clazz = this.tagReplacementClasses[tree.tagName];
                if (parentClasses.indexOf(clazz) != - 1)
                    clazz = "";
                childHTML = this.childrenToHtml(tree, parentClasses + " " + clazz);
                if (childHTML != "")
                    return '<span class="' + clazz + '">' + childHTML + "</span>";
                else
                    return "";
            case "p":
                // check if we're dealing with a list element
                if ("style" in tree.attrs && "mso-list" in tree.attrs.style) {
                    buffer = "";
                    var listLevel = this.list.level;
                    if (matches = tree.attrs.style["mso-list"].match(/^\s*l(\d+)\s+level(\d+)\s+\w+$/)) {
                        listL = matches[1];
                        listLevel = matches[2];
                    }
                    if (listL != this.list.l) {
                        var closer;
                        while (closer = this.list.closers.pop())
                            buffer += closer;
                        this.list.l = listL;
                        this.list.level = 0;
                    }
                    while (listLevel > this.list.level) {
                        ++this.list.level;
                        var opener, closer;
                        opener = '<ol class="style-decimal">';
                        closer = '</ol>';
                        var listRule = this.listRules[this.list.l] && this.listRules[this.list.l][this.list.level];
                        if (!listRule) {} else if (listRule["mso-level-text"] === "o") {
                            opener = '<ul class="style-circle">';
                            closer = "</ul>";
                        } else if (listRule["mso-level-text"]) {
                            opener = "<ul class='style-decimal'>";
                            closer = "</ul>";
                        } else if (listRule["mso-level-number-format"]) {
                            switch (listRule["mso-level-number-format"]) {
                            case "roman-upper":
                                opener = '<ol class="style-upper-roman">';
                                closer = '</ol>';
                                break;
                            case "roman-lower":
                                opener = '<ol class="style-lower-roman">';
                                closer = '</ol>';
                                break;
                            case "alpha-upper":
                                opener = '<ol class="style-upper-alpha">';
                                closer = '</ol>';
                                break;
                            case "alpha-lower":
                                opener = '<ol class="style-lower-alpha">';
                                closer = '</ol>';
                                break;
                            }
                        }
                        buffer += opener;
                        this.list.closers.push(closer);
                    }
                    while (listLevel < this.list.level) {
                        --this.list.level;
                        buffer += this.list.closers.pop();
                    }
                    childHTML = this.childrenToHtml(tree, parentClasses);
                    return buffer + "<li>" + childHTML + "</li>";
                }

            case "ul":
            case "ol":
                if (tree.attrs["class"] && tree.attrs["class"].match(this.allowedListClasses)) {
                    clazz = tree.attrs["class"];
                } else if (tree.tagName == "ul" || tree.tagName == "ol") {
                    clazz = tree.tagName == "ul" ? "style-circle" : "style-text";
                }

                var closer;
                while (closer = this.list.closers.pop())
                    buffer += closer;
                this.list.level = 0;

            case "li":
            case "span":
                if (tree.attrs["style"] && tree.attrs["style"]["mso-list"] == 'Ignore')
                    return "";
            case "div":
            case "blockquote":
                tagHTML = "";
                if (tree.attrs["class"]) {
                    $.each(tree.attrs["class"].split(/\s+/), function(i, clazzItem) {
                        if (clazzItem.match(self.wizzyClassNames) && parentClasses.indexOf(clazzItem) == - 1)
                            clazz += " " + clazzItem;
                    });
                }
                var styles = "";
                if ("style" in tree.attrs) {
                    $.each(tree.attrs.style, function(key, value) {
                        if (key == "font-weight") {
                            if (value == "bold" && clazz.indexOf("wz-bold") == - 1 && parentClasses.indexOf("wz-bold") == - 1) {
                                clazz += " wz-bold";
                            }
                        } else if (key == "font-style") {
                            if (value == "italic" && clazz.indexOf("wz-italic") == - 1 && parentClasses.indexOf("wz-italic") == - 1) {
                                clazz += " wz-italic";
                            }
                        } else if (key == "text-decoration") {
                            if (value.indexOf("underline") != - 1 && clazz.indexOf("wz-underline") == - 1 && parentClasses.indexOf("wz-underline") == - 1) {
                                clazz += " wz-underline";
                            }
                            if (value.indexOf("line-through") != - 1 && clazz.indexOf("wz-strikethrough") == - 1 && parentClasses.indexOf("wz-strikethrough") == - 1) {
                                clazz += " wz-ztrikethrough";
                            }
                        } else if (key.match(self.allowedStyleRules)) {
                            styles += key + ": " + value + "; ";
                        }
                    });
                }
                var startTag, endTag;
                startTag = "", endTag = "";
                if (tree.tagName != "span" || styles != "" || clazz != "" || tree.attrs["id"]) {
                    startTag = '<' + tree.tagName;
                    if (tree.attrs["id"])
                        startTag += " id='" + tree.attrs["id"] + "'";
                    if (styles != "") {
                        if (styles.match(/\"/))
                            startTag += " style='" + styles + "'";
                        else
                            startTag += " style=\"" + styles + "\"";
                    }
                    if (clazz != "")
                        startTag += " class='" + clazz + "'";
                    startTag += ">";
                    endTag = "</" + tree.tagName + ">";
                }
                childHTML = this.childrenToHtml(tree, parentClasses + " " + clazz);
                if ((typeof childHTML === "string" && childHTML.length > 0) || tree.attrs["id"]) {
                    tagHTML += startTag + childHTML + endTag;
                }
                return buffer + tagHTML;

            case "a":
                var url = tree.attrs.href;

                // Whitelist for classes we will allow to be pasted.
                var allowedLinkClasses = [
                "w-link-new-window"
                ];

                // We want to maintain the classes if they are there.
                var linkClasses = (tree.attrs['class'] || "").split(" "),
                newClasses = "",
                classAttr = "";

                // Loop through the whitelist and keep classes we allow.
                for (var i = 0; i < allowedLinkClasses.length; i++) {
                    if (linkClasses.indexOf(allowedLinkClasses[i]) != - 1) {
                        newClasses += allowedLinkClasses[i] + " ";
                    }
                }

                // If any of the classes were whitelisted, construct the class attribute.
                if (newClasses != "") {
                    classAttr = 'class="' + newClasses + '"';
                }

                // If we know abotu this link, reconstruct it, otherwise, make a new
                // linksData entry and construct it.
                if (tree.attrs.id && tree.attrs.id in this.linksData) {
                    return buffer + '<a ' + classAttr + ' id="' + tree.attrs.id + '" href="' + url + '">' + this.childrenToHtml(tree, parentClasses) + "</a>";
                } else if (url && !url.match(/^javascript:/)) {
                    var linkId = Spine.guid();
                    this.linksData[linkId] = {
                        tab: "webpage",
                        subtab: "other-site",
                        url: url
                    };

                    // Maintain the new window property for the editor.
                    if (linkClasses.indexOf("w-link-new-window") != - 1) {
                        this.linksData[linkId].newWindow = true;
                    }
                    return buffer + '<a ' + classAttr + ' href="' + url + '" id="' + linkId + '">' + this.childrenToHtml(tree, parentClasses) + "</a>";
                } else {
                    return buffer + this.childrenToHtml(tree, parentClasses);
                }
                break;
            case "font":
                return this.childrenToHtml(tree, parentClasses);
            default:
                // Keep the children, but don't spit out this unknown tag
                return buffer + this.childrenToHtml(tree, parentClasses);
            }
        },

        childrenToHtml: function(tree, parentClasses) {
            if (!tree || !tree.children || tree.children.length < 1) 
                return "";
            var self = this;
            return $.map(tree.children, function(child) {
                return self.treeToHtml(child, parentClasses);
            }).join("");
        },

        tagReplacementClasses: {
            'b': "wz-bold",
            'i': "wz-italic",
            "u": "wz-underline",
            "s": "wz-strikethrough;"
        },

        allowedListClasses: /^style-(disc|circle|square|decimal|upper-alpha|lower-alpha|upper-roman|lower-roman)$/,
        allowedStyleRules: /^(text-align)$/,
        wizzyClassNames: /^wz-/,

        processStyleTag: function(tree) {
            var self = this;
            $.each(tree.children, function(index, tag) {
                if (!tag || !tag.content) 
                    return;
                var content = tag.content.substr(tag.content.indexOf("@list"));
                var matches;
                while (matches = content.match(/@list\s*l(\d+)(?::level(\d+))?\s*{([^}]*)}\s*([\s\S]*)$/)) {
                    content = matches[4];
                    var l = matches[1],
                    level = matches[2],
                    style = htmlParser.fn.parseStyleRules(matches[3]);
                    (self.listRules[l] || (self.listRules[l] = {}))[level] = style;
                }
            });
        }

    });

    Wizzy.register('paste', function(opts) {
        var w = this;
        function tidyUp() {
            setTimeout(function() {
                w.setDirtyRange();
                w.select(function() {
                    var id = "w-p-id-" + parseInt( + new Date());
                    var range = w.getRange();
                    w.insertHTML("<span id=" + id + "></span>");
                    PasteFromWord.init($(w.el).html(), w.data.linksData).process(function(generatedHTML) {
                        $(w.el).html(generatedHTML);
                        var karet = $(w.el).find("#" + id);
                        range = w.getRange();
                        range.selectNode(karet[0]);
                        w.selectRange(range);
                        w.setDirtyRange();
                        w.select(function() {
                            karet.attr("id", "");
                        });
                    });
                });
            }, 1);
        };
        // Post-process after drop events.
        $(this.el).bind("dragdrop drop", tidyUp);
        if ($.browser.mozilla) {
            // Firefox doesn't give you access to the clipboard, nor does it let you redirect a paste event
            // We should let the paste proceed, and then clean up the entire wizzy.
            $(this.el).bind("paste", tidyUp);
        } else {
            $(w.el).bind("beforepaste", function(e) {
                return false;
            });
            $(w.el).bind("paste", function(e) {
                var sourceHtml = null;
                var range;
                if (e.originalEvent.clipboardData && e.originalEvent.clipboardData.getData) {
                    if (opts.plainText) {
                        sourceHtml = e.originalEvent.clipboardData.getData('text/plain').replace(/\n/g, "<br/>");
                    } else {
                        sourceHtml = e.originalEvent.clipboardData.getData('text/html');
                    }
                    if (sourceHtml && sourceHtml.length > 0) {
                        w.setDirtyRange();
                        w.select(function() {
                            range = w.getRange();
                            PasteFromWord.init(sourceHtml, w.data.linksData).pasteToWizzy(w).done(tidyUp);
                        });
                        return false;
                    }
                }

                var iframe = $("<iframe/>").css({
                    "position": "absolute",
                    "left": "-9999px"
                });
                $(w.el).before(iframe);
                ifrm = iframe[0]
                var pasteTarget = undefined;
                var idoc = ((ifrm.contentWindow) ?
                ifrm.contentWindow :
                (ifrm.contentDocument.document) ?
                ifrm.contentDocument.document :
                ifrm.contentDocument
                ).document;
                idoc.open();
                idoc.write('<body contenteditable="true" designMode="on"></body>');
                idoc.close();
                pasteTarget = idoc.body;

                w.setDirtyRange();
                range = w.getRange();
                pasteTarget.innerHTML = "";
                pasteTarget.focus();
                setTimeout(function() {
                    w.setDirtyRange();
                    w.select(function() {
                        $(w.el).focus();
                        w.selectRange(range);
                        if (opts.plainText) {
                            sourceHtml = pasteTarget.innerText.replace(/\n/g, "<br/>");
                        } else {
                            sourceHtml = pasteTarget.innerHTML;
                        }
                        iframe.remove();
                        PasteFromWord.init(sourceHtml, w.data.linksData).pasteToWizzy(w).done(tidyUp);
                    });
                }, 1);

            });
        }
    });

});
/* jshint ignore:end */
;
/* global require:false, webs:false, jQuery:false */
// initialize canvas within bldr
(function($) {
    "use strict";

    var attempts = 0,
    initCanvasTimer,
    editFrameStyledDeferred = $.Deferred(),
    editFrameStyledPromise = editFrameStyledDeferred.promise();
    function initCanvas() {
        if (parent.bldr && typeof parent.bldr.initPageController === 'function') {
            require(['jquery'], function(jQuery) {
                var pageController = parent.bldr.initPageController({
                    $: jQuery,
                    win: window,
                    doc: document
                });
                pageController.innerFrameStyledPromise.done(editFrameStyledDeferred.resolve).fail(editFrameStyledDeferred.reject);
                clearTimeout(initCanvasTimer);
            });
        } else {
            if (attempts > 10000) {
                // try for 200 seconds
                if (window.console) {
                    window.console.error("Oh Dear, it's taking too long to load, something must've gone wrong");
                }
            } else {
                initCanvasTimer = setTimeout(initCanvas, 200);
                attempts++;
            }
        }
    }
    initCanvas();

    require([ // requirements
    'jquery',
    'internal/sitebuilder/common/eventBuffer',
    'internal/sitebuilder/view/view.app',
    'require',
    'internal/sitebuilder/builder/sitebase',
    'internal/sitebuilder/common/finchUi',
    'internal/sitebuilder/common/creativeCommons',
    'nodeDataTooltip',
    'internal/sitebuilder/common/ModuleClassLoader',
    // jquery plugins
    'jquery.watch',
    'jquery.customRange',
    'jquery.numberInput',
    'internal/sitebuilder/common/jquery.loadingSpinner',
    // image controls
    'jquery.cropper',
    'jquery.dragger',
    'jquery.panner',
    'jquery.resizer',
    'jquery.zoomer',
    // wizzy controls
    'internal/sitebuilder/builder/wizzy',
    'internal/sitebuilder/builder/wizzy.ui',
    'internal/sitebuilder/builder/wizzy.paste'
    ], function bootstrap_initEditMode($, eventBuffer) {

        $(function() {
            editFrameStyledPromise.done(function() {
                webs.fixNavWrap();

                // In builder, add a distinguishing class to the HTML
                $('html').addClass('webs-mode-edit');

                var loadSecs = (new Date() - top.webs.startLoad) / 1000;
                eventBuffer.trigger("loaded:sitebuilder", {
                    loadSecs: loadSecs 
                });
            });
        });
    });
})(jQuery);

define("internal/sitebuilder/builder/builder.app", function() {});

