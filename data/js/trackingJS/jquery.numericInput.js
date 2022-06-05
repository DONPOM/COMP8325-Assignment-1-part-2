(function(d) {
    function h() {
        d.isFunction(this.init) && this.init.apply(this, arguments)
    }
    d.fn.numericInput = function(a) {
        var c = Array.prototype.slice.call(arguments), b = c.shift();
        return this.each(function(e, f) {
            var g = d.data(f, "numericInput") || d.data(f, "numericInput", new h(f, a));
            b && "string" === typeof b && "_" !== b.charAt(0) && d.isFunction(g[b]) && g[b].apply(g, c)
        })
    };
    d.fn.numericInput.defaults = {
        change: null,
        decimals: 0,
        decimalSymbol: "",
        disabled: !1,
        maxValue: 1E6,
        minValue: 0,
        mouseHold: !0,
        present: null,
        steps: 1,
        thousandSymbol: ""
    };
    var f;
    h.prototype = {
        init: function(a, c) {
            var b = d.fn.numericInput.defaults, e;
            this.$element = d(a);
            this.$input = this.$element.find("input:first");
            this.$label = this.$element.closest("label");
            e = {
                decimals: parseInt(this.$input.data("decimals"), 10) || b.decimals,
                decimalSymbol: this.$input.data("decimal-symbol") || b.decimalSymbol,
                disabled: this.$input.is(":disabled"),
                maxValue: parseInt(this.$input.data("max-value"), 10) || b.maxValue,
                minValue: parseInt(this.$input.data("min-value"), 10) || b.minValue,
                mouseHold: this.$input.data("mouse-hold") ||
                b.mouseHold,
                steps: parseInt(this.$input.data("steps"), 10) || b.steps,
                thousandSymbol: this.$input.data("thousand-symbol") || b.thousandSymbol
            };
            this.options = d.extend({}, b, e, c);
            this.value = this.options.minValue;
            this.value = this._normalizeAndReturnNumericValue(this.$input.val());
            this.setInputWidth();
            this._presentValue(this.value);
            this._bindEventsToElement();
            this.options.disabled ? this.disable() : this.enable()
        },
        enable: function() {
            this.$element.actionController("enable");
            this.$label.removeClass("cc-state-disabled");
            this.$input.prop("disabled", !1)
        },
        disable: function() {
            this.$element.actionController("disable");
            this.$label.addClass("cc-state-disabled");
            this.$input.prop("disabled", !0)
        },
        getValue: function() {
            return this.value
        },
        setMinimum: function(a) {
            this.options.minValue = parseInt(a, 10)
        },
        getMinimum: function() {
            return this.options.minValue
        },
        setMaximum: function(a) {
            this.options.maxValue = parseInt(a, 10)
        },
        getMaximum: function() {
            return this.options.maxValue
        },
        setInputWidth: function() {
            var a = this.options.maxValue + "";
            2 > a.length &&
            (a = Array(3).join("0"));
            a = this._formatDecimals(a);
            this.$input.attr("maxlength", a.length)
        },
        _getMaximumInnerWidth: function(a) {
            a = ("" + a).length;
            var c = this.$input.css("font-size");
            f || (c = d('\x3cspan id\x3d"cc-numeric-input-font-size" style\x3d"display: none; font-size: ' + c + ';"\x3e0\x3c/span\x3e').appendTo("body"), f = c.width(), c.remove());
            a*=f;
            return "border-box" === this.$input.css("box-sizing") ? a + (this.$input.outerWidth() - this.$input.width()) : a
        },
        changeValue: function(a) {
            this.value = this._normalizeAndReturnNumericValue(a);
            this._presentValue(this.value);
            d.isFunction(this.options.change) && this.options.change.call(this.$input[0], this.value)
        },
        modifyValue: function(a) {
            this._modifyValueByDirection(a)
        },
        _presentValue: function(a) {
            var c = a;
            a = Infinity === a ? "\u221e" : this._formatDecimals(a);
            d.isFunction(this.options.present) && (a = this.options.present.call(this.$input[0], a, c));
            this.$input.val(a)
        },
        _bindEventsToElement: function() {
            this.$element.actionController(this, {
                events: "click change mousedown mouseup mouseout keydown keyup",
                context: "controller"
            })
        },
        _arrowMousedown: function(a, c) {
            var b = this;
            d("body").one("mouseup", function() {
                b._clearMouseHoldDefer()
            });
            this._modifyValueByDirection(c, 300)
        },
        _arrowMouseup: function() {
            this._clearMouseHoldDefer()
        },
        _arrowMouseout: function() {
            this._clearMouseHoldDefer()
        },
        _inputKeyup: function() {
            this._clearMouseHoldDefer()
        },
        _inputChange: function() {
            this.changeValue(this.$input.val())
        },
        _inputKeydown: function(a) {
            switch (a.keyCode) {
            case 38:
                a = "up";
                break;
            case 40:
                a = "down";
                break;
            case 13:
            case 10:
                return this.changeValue(this.$input.val()),
                !0;
            default:
                return !0
            }
            this._modifyValueByDirection(a)
        },
        _infiniteClick: function(a) {
            a.preventDefault();
            this.changeValue(Infinity)
        },
        _modifyValueByDirection: function(a, c) {
            var b = this, e = 0;
            this.$input.is(":disabled") || (this.value = this._normalizeAndReturnNumericValue(this.$input.val()), "down" === a ? e =- 1 * this.options.steps : "up" === a && (e = 1 * this.options.steps), e = Infinity === this.value ? this.options.minValue : this.value + e, this.changeValue(e), this.options.mouseHold && c && (this.mouseHoldDefer = d.defer(function() {
                b._modifyValueByDirection(a,
                50)
            }, c)))
        },
        _normalizeAndReturnNumericValue: function(a) {
            var c = this.options.minValue, b = this.options.maxValue;
            if (Infinity === a || "\u221e" === a)
                return Infinity;
            if ("" === a)
                return Math.max(0, c);
            a = this._parseNumeric(a);
            return isNaN(a) ? this.value : a < c ? c : b && a > b ? b : a
        },
        _parseNumeric: function(a) {
            return "number" === typeof a ? a : 0 < this.options.decimals ? parseFloat(this._removeThousandSymbols(a).replace(this.options.decimalSymbol, ".")) : parseInt(this._removeThousandSymbols(a), 10)
        },
        _formatDecimals: function(a) {
            if (0 < this.options.decimals) {
                a =
                (a + "").split(".");
                var c = a[1] || "", b = Math.max(this.options.decimals - c.length + 1, 1);
                return a[0] + this.options.decimalSymbol + c.substr(0, 2) + Array(b).join("0")
            }
            return a
        },
        _clearMouseHoldDefer: function() {
            d.isFunction(this.mouseHoldDefer) && this.mouseHoldDefer()
        },
        _removeThousandSymbols: function(a) {
            return this.options.thousandSymbol ? (a + "").split(this.options.thousandSymbol).join("") : a
        }
    }
})(jimdoGen002);
