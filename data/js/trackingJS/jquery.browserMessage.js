(function(d) {
    function g() {
        d.isFunction(this.init) && this.init.apply(this, arguments)
    }
    g.prototype = {
        init: function(a, b) {
            var c = {};
            this.defaults = d.fn.browserMessage.defaults;
            d.extend(!0, c, this.defaults, a);
            this.options = c;
            this.$element = d(b);
            this.$textNode = this.$closeButton = this.$messageElement = void 0;
            c && c.tests && c.autoRun && this._runAllTests()
        },
        destroy: function() {
            var a = this._getMessageElement(), b = this.$element;
            this._clearAutoHideTimeout();
            this._unbindAll();
            a.remove();
            b.removeData("browserMessage");
            b.trigger("destroy.browserMessage")
        },
        test: function(a) {
            var b=!1, c = this.options;
            c && c.tests && c.tests[a] && (a = c.tests[a], d.isFunction(a.test) && (b = a.test.apply(this.$element)) && this.show(a.message));
            this.$element.trigger("test.browserMessage");
            return b
        },
        show: function(a) {
            var b = this._getMessageElement(), c = this.$element, d = this.options, e = this._getMessageAsString(a);
            b.length || (b = this._createMessageElement());
            e && this._getTextNode().html(a);
            b.length && (this._messageBoxEffectShow(), d.autoHide && this._startAutoHideTimeout(d.autoHide));
            c.trigger("show.browserMessage")
        },
        hide: function() {
            var a = this.$element;
            this._clearAutoHideTimeout();
            this._messageBoxEffectHide();
            a.trigger("hide.browserMessage")
        },
        _getMessageAsString: function(a) {
            var b = this.$element;
            return d.isFunction(a) ? a.apply(b) : a
        },
        _messageBoxEffectShow: function() {
            var a = this, b = a._getMessageElement(), c = a._getCloseButton();
            b.length && (c.show(), b.stop().css("margin-top", - 1 * b.height()).show().animate({
                "margin-top": 0
            }, 500, "linear", function() {
                a._scrollToElement(b)
            }))
        },
        _messageBoxEffectHide: function() {
            var a = this._getMessageElement(),
            b = this._getCloseButton();
            a.length && (b.hide(), a.stop().animate({
                "margin-top": - 1 * a.height()
            }, 500, "linear", function() {
                a.hide()
            }))
        },
        _scrollToElement: function(a) {
            this._isNotInViewport(a) && d("html,body").animate({
                scrollTop: a.offset().top
            }, "fast", "linear")
        },
        _isNotInViewport: function(a) {
            var b = d(document).scrollTop();
            if (a.length)
                return b >= a.offset().top + a.height()
        },
        _clearAutoHideTimeout: function() {
            this.timeOutId && (clearTimeout(this.timeOutId), delete this.timeOutId)
        },
        _startAutoHideTimeout: function(a) {
            var b =
            this, c = b.defaults;
            a = parseInt(a, 10);
            a || (a = c.autoHide);
            b._clearAutoHideTimeout();
            b.timeOutId = setTimeout(function() {
                b.hide()
            }, a)
        },
        _runAllTests: function() {
            var a, b = this.options, c = b.tests, d = Object.prototype.hasOwnProperty;
            if (b && c)
                for (a in c)
                    if (d.call(c, a) && this.test(a))
                        break
        },
        _buildHtml: function(a) {
            return d.mustache(this.options.template, {
                message: a,
                messageClass: this.options.messageClass,
                iconClass: this.options.iconClass,
                textClass: this.options.textClass,
                closeClass: this.options.closeClass,
                sliceClass: this.options.sliceClass,
                containerClass: this.options.containerClass,
                backgroundClass: this.options.backgroundClass
            })
        },
        _createMessageElement: function() {
            this.$element.prepend(this._buildHtml());
            this._bindEventsToCloseButton();
            return this._getMessageElement()
        },
        _bindEventsToCloseButton: function() {
            var a = this, b = a._getCloseButton(), c = a.options.hoverClass;
            b.bind("click.browserMessage mouseover.browserMessage mouseout.browserMessage", function(d) {
                d.preventDefault();
                switch (d.type) {
                case "click":
                    a.hide();
                    break;
                case "mouseover":
                    b.addClass(c);
                    break;
                case "mouseout":
                    b.removeClass(c)
                }
            })
        },
        _unbindAll: function() {
            this._getCloseButton().unbind(".browserMessage")
        },
        _getTextNode: function() {
            var a = this.$textNode;
            a && a.length || (a = this._getMessageElement().find("." + this.options.textClass + ":first"));
            return a
        },
        _getMessageElement: function() {
            var a = this.$messageElement;
            a && a.length || (a = this.$element.find("." + this.options.messageClass + ":first"));
            return a
        },
        _getCloseButton: function() {
            var a = this.$closeButton;
            a && a.length || (a = this._getMessageElement().find("." + this.options.closeClass +
            ":first"));
            return a
        }
    };
    d.fn.extend({
        browserMessage: function(a) {
            var b = Array.prototype.slice.call(arguments), c = b.shift();
            return this.each(function(h, e) {
                var f = d.data(e, "elementOverlays") || d.data(e, "elementOverlays", new g(a, e));
                c && "string" === typeof c && "_" !== c.charAt(0) && d.isFunction(f[c]) && f[c].apply(f, b)
            })
        }
    });
    d.fn.browserMessage.defaults = {
        template: '\x3cdiv class\x3d"{{messageClass}} "\x3e                       \x3cdiv class\x3d"{{containerClass}} {{sliceClass}} {{backgroundClass}}"\x3e                           \x3cdiv class\x3d"{{sliceClass}} {{iconClass}}"\x3e \x3c/div\x3e                           \x3cdiv class\x3d"{{textClass}}"\x3e\x3c/div\x3e                           \x3cdiv class\x3d"{{sliceClass}} {{closeClass}}"\x3e\x3c/div\x3e                        \x3c/div\x3e                   \x3c/div\x3e',
        messageClass: "cc-browser-message",
        iconClass: "cc-browser-icon",
        textClass: "cc-browser-text",
        closeClass: "cc-browser-close",
        hoverClass: "cc-state-hover",
        sliceClass: "cc-browser-slice",
        containerClass: "cc-browser-container",
        backgroundClass: "cc-browser-slice-bg",
        tests: [],
        autoRun: !0,
        autoHide: 15E3
    }
})(jimdoGen002);
