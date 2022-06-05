/*!CK:3152292121!*/
/*1413168435,*/

if (self.CavalryLogger) {
    CavalryLogger.start_js(["fmU\/P"]);
}

__d("XFriendJewelNotifsControllerURIBuilder", ["XControllerURIBuilder"], function(a, b, c, d, e, f) {
    e.exports = b("XControllerURIBuilder").create("\/friendjewel\/friendconfirmednotifications\/", {
        alert_ids: {
            type: "IntVector",
            required: true
        },
        action: {
            type: "String",
            required: true
        }
    });
}, null);
__d("FriendConfirmedNotifs", ["Event", "shield", "CSS", "DOM", "ge", "DataStore", "Arbiter", "AsyncRequest", "Parent", "XFriendJewelNotifsControllerURIBuilder"], function(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) {
    function q(r) {
        "use strict";
        this.$FriendConfirmedNotifs0 = r.aggregated;
        var s = r.close;
        for (var t = 0; t < s.length; ++t)
            g.listen(s[t], 'click', this.$FriendConfirmedNotifs1.bind(this));
        this.$FriendConfirmedNotifs2 = r.unread_notifs;
        this.$FriendConfirmedNotifs3 = r.unseen_notifs;
        this.$FriendConfirmedNotifs4 = r.has_mark_seen;
        this.$FriendConfirmedNotifs5 = [];
        if (this.$FriendConfirmedNotifs2.length)
            for (var u = 0; u < r.units.length; ++u)
                this.$FriendConfirmedNotifs5[u] = g.listen(r.units[u], 'click', this.$FriendConfirmedNotifs6.bind(this, u));
        if (this.$FriendConfirmedNotifs3.length && this.$FriendConfirmedNotifs4 || this.$FriendConfirmedNotifs2.length&&!this.$FriendConfirmedNotifs4)
            this.$FriendConfirmedNotifs7 = m.subscribe('friend-confirmed-notifs-seen', h(this.$FriendConfirmedNotifs8, this));
        if (r.read_notifs_to_set.length)
            this.$FriendConfirmedNotifs9(r.read_notifs_to_set, 'set_time');
        this.$FriendConfirmedNotifsa = r.notifids;
    }
    q.prototype.$FriendConfirmedNotifsb = function(r) {
        "use strict";
        var s = false, t = k('fbRequestsFlyout'), u = j.find(t, '.requestsUnitTitle'), v = j.scry(r, '.friendConfirmedNotifsUnitTitle')[0];
        if (!v) {
            s = true;
            v = j.find(t, '.friendConfirmedNotifsUnitTitle');
        }
        if (s) {
            i.show(j.find(u, '.requestsJewelLinks'));
            j.replace(v, u);
        } else 
            i.hide(v);
    };
    q.prototype.$FriendConfirmedNotifs1 = function(event) {
        "use strict";
        var r = event.getTarget(), s, t = [];
        if (this.$FriendConfirmedNotifs0) {
            var u = o.byClass(r, 'friendConfirmedNotifsUnitAggregated');
            t = l.get(u, 'notifids');
            s = u.parentElement;
            i.hide(u);
        } else {
            var v = o.byClass(r, 'friendConfirmedNotifsUnitDisaggregated');
            t = l.get(v, 'notifids');
            s = v.parentElement.parentElement;
            i.hide(v);
        }
        for (var w = 0; w < t.length; ++w) {
            var x = this.$FriendConfirmedNotifsa.indexOf(t[w]);
            if (x>-1)
                this.$FriendConfirmedNotifsa.splice(x, 1);
        }
        if (!this.$FriendConfirmedNotifsa.length)
            this.$FriendConfirmedNotifsb(s);
        this.$FriendConfirmedNotifs9(t, 'xout');
    };
    q.prototype.$FriendConfirmedNotifs9 = function(r, s) {
        "use strict";
        var t = new p().setIntVector('alert_ids', r).setString('action', s).getURI();
        new n().setURI(t).send();
    };
    q.prototype.$FriendConfirmedNotifs8 = function() {
        "use strict";
        var r = this.$FriendConfirmedNotifs4 ? 'mark_seen': 'mark_read';
        this.$FriendConfirmedNotifs7.unsubscribe();
        this.$FriendConfirmedNotifs9(this.$FriendConfirmedNotifs3, r);
    };
    q.prototype.$FriendConfirmedNotifs6 = function(r, event) {
        "use strict";
        var s = event.getTarget(), t = o.byClass(s, 'friendConfirmedNotifsUnread');
        i.removeClass(t, 'friendConfirmedNotifsUnread');
        var u = [];
        if (this.$FriendConfirmedNotifs0) {
            var v = o.byClass(t, 'friendConfirmedNotifsUnitAggregated');
            u = l.get(v, 'notifids');
        } else {
            var w = o.byClass(t, 'friendConfirmedNotifsUnitDisaggregated');
            u = l.get(w, 'notifids');
        }
        this.$FriendConfirmedNotifs5[r].remove();
        if (this.$FriendConfirmedNotifs4)
            this.$FriendConfirmedNotifs9(u, 'mark_read');
    };
    f.init = function(r) {
        new q(r);
    };
}, null);
__d("FriendConfirmedNotifsClickLogging", ["Event", "AsyncSignal"], function(a, b, c, d, e, f, g, h) {
    var i = {
        addClickLoggingListener: function(j, k) {
            g.listen(j, 'click', function() {
                new h(k).send();
            });
        }
    };
    e.exports = i;
}, null);
__d("JewelFollowupUnit", ["DOM", "Arbiter", "CSS", "csx", "copyProperties"], function(a, b, c, d, e, f, g, h, i, j, k) {
    function l(m, n, o, p) {
        this._followup = n;
        this._parentContainer = this._followup.parentNode;
        this._friendRequest = this._parentContainer.firstChild;
        o && h.subscribe(o, function(r, s) {
            (s.uid == m) && this._showFollowup();
        }.bind(this));
        var q = g.scry(this._parentContainer, "._e7")[0];
        q && (q.onclick = this._hideFollowup.bind(this));
        p && h.subscribe(p, function(r, s) {
            (s.uid == m) && this._hideFollowup();
        }.bind(this));
    }
    k(l.prototype, {
        _showFollowup: function() {
            this._updateRequestAnimClass();
            i.addClass(this._followup, 'active');
            i.addClass(this._parentContainer, 'followupActivated');
        },
        _hideFollowup: function() {
            this._updateRequestAnimClass();
            i.removeClass(this._followup, 'active');
            i.removeClass(this._parentContainer, 'followupActivated');
        },
        _updateRequestAnimClass: function() {
            i.conditionClass(this._friendRequest, 'animated', i.hasClass(this._followup, 'animated'));
        }
    });
    e.exports = l;
}, null);
__d("InnerStickyArea", ["ContextualLayer", "CSS", "DataStore", "DOM", "Event", "LayerHideOnTransition", "Locale", "Parent", "Style", "Vector", "cx", "removeFromArray"], function(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r) {
    function s(v) {
        var w = n.byClass(v, 'scrollable') || o.getScrollParent(v.parentNode);
        return w;
    }
    function t(v) {
        "use strict";
        var w = s(v);
        this.node = v;
        this._extracted = false;
        this._placeholder = j.create('div', {
            className: "_ptr"
        });
        u.getInstance(w).register(this);
    }
    t.prototype.update = function() {
        "use strict";
        if (this._extracted) {
            p.getElementDimensions(this._placeholder).setElementWidth(this.node);
            p.getElementDimensions(this.node).setElementHeight(this._placeholder);
        } else 
            p.getElementDimensions(this.node).setElementWidth(this.node).setElementHeight(this._placeholder);
        return this;
    };
    t.prototype.setExtracted = function(v) {
        "use strict";
        if (v === this._extracted)
            return this;
        if (v) {
            this.update();
            j.replace(this.node, this._placeholder);
        } else {
            o.set(this.node, 'height', null);
            o.set(this.node, 'width', null);
            if (this._placeholder.parentNode) {
                j.replace(this._placeholder, this.node);
            } else 
                j.remove(this.node);
        }
        this._extracted = v;
        return this;
    };
    t.prototype.getInlineNode = function() {
        "use strict";
        return this._extracted ? this._placeholder : this.node;
    };
    t.prototype.isDisplayed = function() {
        "use strict";
        var v = this.getInlineNode();
        return v.offsetWidth > 0 && v.offsetHeight > 0;
    };
    t.prototype.getStickyContainer = function() {
        "use strict";
        return this._stickyContainer;
    };
    t.prototype.updateContainer = function() {
        "use strict";
        var v = s(this.node);
        u.getInstance(v).register(this);
    };
    t.prototype.destroy = function() {
        "use strict";
        this.getStickyContainer().unregister(this);
    };
    t.getStickyContainer = function(v) {
        "use strict";
        return u.getInstance(s(v));
    };
    function u(v) {
        "use strict";
        this.node = v;
        this._areas = [];
        this._fixTarget = null;
        this._fixedArea = null;
        this._initialized = false;
        this._layer = new g({
            permanent: true
        }, j.create('div')).setInsertParent(this.node.parentNode).disableBehavior(l);
        this._listener = k.listen(v, 'scroll', this.update.bind(this));
        h.addClass(v, "_pts");
        i.set(v, 'StickyContainer', this);
    }
    u.prototype.isDisplayed = function() {
        "use strict";
        return this.node.offsetWidth > 0 && this.node.offsetHeight > 0;
    };
    u.prototype.register = function(v) {
        "use strict";
        if (v.getStickyContainer())
            v.getStickyContainer().unregister(v);
        v._stickyContainer = this;
        this._areas.push(v);
        this.update();
        return this;
    };
    u.prototype.unregister = function(v) {
        "use strict";
        r(this._areas, v);
        this.update();
    };
    u.prototype.update = function() {
        "use strict";
        if (!this.isDisplayed())
            return this;
        var v = null, w = this, x = this.node.scrollTop, y;
        for (var z = 0; z < this._areas.length; z++) {
            var aa = this._areas[z], ba = aa.getInlineNode();
            if (!aa.isDisplayed())
                continue;
            if (!j.contains(this.node.parentNode, ba))
                continue;
            var ca = ba.offsetTop;
            if (ca <= x) {
                if (y === undefined || ca > y) {
                    v = aa;
                    y = ca;
                }
            } else if (v) {
                var da = p.getElementDimensions(v.node).y;
                if (ca - da < x)
                    w = aa;
                break;
            }
        }
        if (this._fixedArea === v && this._fixTarget === w) {
            this._fixedArea && this._fixedArea.update();
        } else {
            if (this._fixedArea && this._fixedArea !== v)
                this._unfixArea(this._fixedArea);
            if (v)
                this._fixAreaTo(v, w);
            this._fixedArea = v;
            this._fixTarget = w;
        }
        return this;
    };
    u.prototype.destroy = function() {
        "use strict";
        this._listener && this._listener.remove();
        this._listener = null;
    };
    u.prototype._fixAreaTo = function(v, w) {
        "use strict";
        this._layer.hide();
        v.setExtracted(true);
        if (w instanceof u) {
            this._layer.setInsertParent(this.node.parentNode).setAlignment(m.isRTL() ? 'right' : 'left').setContext(this.node);
        } else 
            this._layer.setInsertParent(this.node).setContext(w.node);
        this._layer.setContent(v.node).show();
        h.addClass(v.node, "_57kj");
    };
    u.prototype._unfixArea = function(v) {
        "use strict";
        this._layer.hide();
        v.setExtracted(false);
        h.removeClass(v.node, "_57kj");
    };
    u.getInstance = function(v) {
        "use strict";
        var w = i.get(v, 'StickyContainer');
        return w || new u(v);
    };
    e.exports = t;
}, null);
