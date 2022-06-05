

  P.when('A').register('SgHover', function(A) {
    return function(root) {
      var self = this,
          modules = {},
          dragging = false;

      self.register = function(type, module) {
        modules[type] = module;
      };

      if (A.capabilities.touch) {
        return;
      }

      A.$(root).each(function(i, r) {
        A.$(r).delegate('*[data-sghover]', 'mouseenter mouseleave click touchend', function(e) {
          self.handle.call(this, { data: A.$(this).data('sghover'), event: e, target: this, root: r });
        });
      });

      A.$('body').mousedown(function() { dragging = true; }).mouseup(function() { dragging = false; });

      self.handle = function(e) {
        var module = modules[e.data.type];
        var asin = e.data.asin;
        if(dragging || !module || !asin) return;

        if( e.event.type === 'mouseenter' ) {
          _clearTitles(this);
          module['in'].call(this, e);
        } else {
          module['out'].call(this, e);
        }
      }
    }

    function _clearTitles(e) {
      A.$(e).find('[title]').each(function(i, v) {
        A.$(v).removeAttr('title');
      });
    }
  });

  P.when('A','dombinder').register('SgHoverHelper', function(A, binder) {
    return function(hover, pdb) {
      var self = this;

      self.register = function(type, $root, callbacks, options) {
        var showTimer = null,
            loadTimer = null,
            product = null,
            productObserver = null,
            opts = options || {},
            showDelay = opts['showDelay'] || 0,
            loadDelay = opts['loadDelay'] || 0,
            ctx = binder.init($root);

        hover.register(type, {
          'in': function(e) {
            showTimer = setTimeout(function() { _in(e); }, showDelay);
          },
          'out': function(e) {
            _unobserve();
            _hide(e);
          }
        });

        if(opts['disablePointerEvents']) {
          $root.css('pointer-events', 'none');
          $root.bind("tap click mousedown mouseup mouseenter mouseleave", function(e) {
            var target;
            if(document.msElementsFromPoint) {
              var targets = document.elementFromPoint(e.clientX, e.clientY);
              if(targets.length > 1) {
                target = targets[1];
              }
            } else {
              $(this).hide();
              target = document.elementFromPoint(e.clientX, e.clientY);
              $(this).show();
            }
            $(target).trigger(e.type);
            return false;
          });
        }

        function _in(e) {
          if(!e.data) {
            return
          }
          _unobserve();
          product = pdb.get(e.data['asin']);
          binder.update(ctx, product);
          _observe(e);
          loadTimer = setTimeout(function() { _call(callbacks['show'], e); }, loadDelay);
        }

        function _observe(e) {
          _unobserve();
          productObserver = function() {
            binder.update(ctx, product);
            if(!product.loading) {
              _show(e);
            }
          }
          product.pdb_observe(productObserver);
        }

        function _unobserve() {
          if(product && productObserver) {
            product.pdb_unobserve(productObserver);
            productObserver = null;
          }
        }

        function _show(e) {
          _clearDelays();
          _call(callbacks['show'], e);
        }

        function _hide(e) {
          _clearDelays();
          _call(callbacks['hide'], e);
        }

        function _clearDelays() {
          if(loadTimer) {
            clearTimeout(loadTimer);
            loadTimer = null;
          }
          if(showTimer) {
            clearTimeout(showTimer);
            showTimer = null;
          }
        }
      };
    };

    function _call(func, data) {
      if(func) {
        func(data);
      }
    }
  });

  P.when('A').register('dombinder', function(A) {
    function _getProp(obj, path) {
      for (var i = 0, path = path.split('.'), len = path.length; i < len; i++) {
        obj = obj && obj[path[i]];
      }
      return obj;
    }

    var bindingHandlers = {
      'text': {
        update: function(e, v, d) {
          A.$(e).text(_getProp(d, v) || '');
        }
      },
      'trimText': {
        update: function(e, v, d) {
          var $e = A.$(e),
              txt = $e.text();
          if(txt.length > v) {
            $e.text(txt.substring(0, v - 4) + '...');
          }
        }
      },
      'visible': {
        update: function(e, v, d) {
          var negate = false;
          if(v.charAt(0) === '!') {
            negate = true;
            v = v.substring(1);
          }
          if(negate != !!_getProp(d, v)) {
            A.$(e).show();
          } else {
            A.$(e).hide();
          }
        }
      },
      'css': {
        init: function(e, v) {
          var $e = A.$(e),
              cssState = $e.data('dombcss') || [];
          cssState[v] = '';
          $e.data('dombcss', cssState);
        },
        update: function(e, v, d) {
          var $e = A.$(e),
              cssState = $e.data('dombcss'),
              oldClass = cssState[v],
              newClass = _getProp(d, v);
          if(oldClass !== newClass) {
            cssState[v] = newClass;
            if(oldClass) { $e.removeClass(oldClass); }
            if(newClass) { $e.addClass(newClass); }
          }
        }
      }
    };

    function _parse(e) {
      return A.$.map(A.$(e).data('bind').split(','), function(s) {
        var tmp = s.split(':'),
            k = A.$.trim(tmp[0]), v = A.$.trim(tmp[1]),
            b = bindingHandlers[k];
        if(!b) { return; }

        if(b.init) {
          b.init(e, v);
        }
        return function(data) {
          b.update(e, v, data);
        };
      });
    }

    return {
      init: function($root) {
        var ctx = { $root: $root, bindings: [] };
        $root.find('[data-bind]').each(function() {
          A.$.each(_parse(this), function(i, v) {
            ctx.bindings.push(v);
          });
        });
        return ctx;
      },
      update: function(ctx, data) {
        A.$.each(ctx.bindings, function(i, v) {
          v(data);
        });
      }
    };
  });

  P.when('A').register('gw-productdb', function(A) {
    var db = {};

    return {
      _add: function(data) {
        if(data.p) {
          A.$.each(data.p, function(i, p) {
            if(!p.asin) return;
            var entry = db[p.asin] || _new(p.asin);
            A.$.extend(entry, p, { loading: false, error: false });
            entry._trigger(p);
          });
        }
      },
      get: function(asin) {
        if(asin in db) {
          return db[asin];
        } else {
          return _new(asin);
        }
      }
    };

    function _new(asin) {

      var observers = [];

      db[asin] = {
        loading: true,
        pdb_observe: function(handler) {
          observers.push(handler);
        },
        pdb_unobserve: function(handler) {
          observers = A.$.grep(observers, function(v) {
            return v != handler;
          });
        },
        _trigger: function() {
          A.$.each(observers, function() {
            this.call(db[asin]);
          });
        }
      };

      return db[asin];
    }

    function _call(func, data) {
      if(func) {
        func(data);
      }
    }
  });

