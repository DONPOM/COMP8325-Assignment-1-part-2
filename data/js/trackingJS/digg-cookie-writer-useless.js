
    // From https://developer.mozilla.org/en-US/docs/Web/API/document.cookie
      /*\
      |*|
      |*|  :: cookies.js ::
      |*|
      |*|  A complete cookies reader/writer framework with full unicode support.
      |*|
      |*|  https://developer.mozilla.org/en-US/docs/DOM/document.cookie
      |*|
      |*|  This framework is released under the GNU Public License, version 3 or later.
      |*|  http://www.gnu.org/licenses/gpl-3.0-standalone.html
      |*|
      |*|  Syntaxes:
      |*|
      |*|  * docCookies.setItem(name, value[, end[, path[, domain[, secure]]]])
      |*|  * docCookies.getItem(name)
      |*|  * docCookies.removeItem(name[, path], domain)
      |*|  * docCookies.hasItem(name)
      |*|  * docCookies.keys()
      |*|
      \*/

      var docCookies = {
        getItem: function (sKey) {
          return decodeURIComponent(document.cookie.replace(new RegExp("(?:(?:^|.*;)\\s*" + encodeURIComponent(sKey).replace(/[\-\.\+\*]/g, "\\$&") + "\\s*\\=\\s*([^;]*).*$)|^.*$"), "$1")) || null;
        },
        setItem: function (sKey, sValue, vEnd, sPath, sDomain, bSecure) {
          if (!sKey || /^(?:expires|max\-age|path|domain|secure)$/i.test(sKey)) { return false; }
          var sExpires = "";
          if (vEnd) {
            switch (vEnd.constructor) {
              case Number:
                sExpires = vEnd === Infinity ? "; expires=Fri, 31 Dec 9999 23:59:59 GMT" : "; max-age=" + vEnd;
                break;
              case String:
                sExpires = "; expires=" + vEnd;
                break;
              case Date:
                sExpires = "; expires=" + vEnd.toUTCString();
                break;
            }
          }
          document.cookie = encodeURIComponent(sKey) + "=" + encodeURIComponent(sValue) + sExpires + (sDomain ? "; domain=" + sDomain : "") + (sPath ? "; path=" + sPath : "") + (bSecure ? "; secure" : "");
          return true;
        },
        removeItem: function (sKey, sPath, sDomain) {
          if (!sKey || !this.hasItem(sKey)) { return false; }
          document.cookie = encodeURIComponent(sKey) + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT" + ( sDomain ? "; domain=" + sDomain : "") + ( sPath ? "; path=" + sPath : "");
          return true;
        },
        hasItem: function (sKey) {
          return (new RegExp("(?:^|;\\s*)" + encodeURIComponent(sKey).replace(/[\-\.\+\*]/g, "\\$&") + "\\s*\\=")).test(document.cookie);
        },
        keys: /* optional method: you can safely remove it! */ function () {
          var aKeys = document.cookie.replace(/((?:^|\s*;)[^\=]+)(?=;|$)|^\s*|\s*(?:\=[^;]*)?(?:\1|$)/g, "").split(/\s*(?:\=[^;]*)?;\s*/);
          for (var nIdx = 0; nIdx < aKeys.length; nIdx++) { aKeys[nIdx] = decodeURIComponent(aKeys[nIdx]); }
          return aKeys;
        }
      };

      // From https://stackoverflow.com/questions/11582512/how-to-get-url-parameters-with-javascript
      function getURLParameter(name) {
        return decodeURIComponent((new RegExp('[?|&]' + name + '=' + '([^&;]+?)(&|#|;|$)').exec(location.search)||[,""])[1].replace(/\+/g, '%20'))||null
      }

      function createUrl (str, options) {
        if (!str.match(/^http/))
          str = '//' + str;
        var keys = Object.keys(options);
        if (!keys.length)
          return str;
        if (str.indexOf('?') === str.length-1) {
          str = str.slice(0, str.length-1);
        }
        var qs = "?";
        var n, nn;
        for (n=0, nn=keys.length; n<nn; n++) {
          if (n === (nn-1)) { // we are at the final parameter
            qs += keys[n] + '=' + options[keys[n]];
          }
          else {
            qs += keys[n] + '=' + options[keys[n]] + '&';
          }
        }
        return str + qs;
      }

      var removePixel = function(pixel, i, cb) {
        i = i || 0;
        if(pixel.complete || i == 1000) {
            // remove the pixel
            document.body.removeChild(pixel);
            if (cb) {
                cb();
            }
        } else {
            setTimeout(function() {
                removePixel(pixel, i+1,cb);
            }, 100); // wait a bit and try to remove the pixel again
        }
      };

      // Only enable logging if debug=true
      var debug = getURLParameter('debug') || '';
      if (!(debug === 'true' || debug === '1')) {
        console = function() {};
        console.log = function() {};
      }

      // checkAndSetUID is only called if cookieable is true
      // so the user is not cookied, we know a call to cset will succeed
      var checkAndSetUID = function () {
          if (!docCookies.hasItem('uid')) {
              var base      = encodeURIComponent("http://eb2.3lift.com/cset?key=uid&value=$UID");
              var adnxsPart = "//ib.adnxs.com/getuid?"+base;
              var i         =  document.createElement('IMG');
              i.src         = adnxsPart;
              i.display     = "none";
              i.width       = 0;
              i.height      = 0;
              document.body.appendChild(i);
              removePixel(i,0,function () { // this call back is to set a uu cookie for backwards compatibility with eb1
                  var base      = encodeURIComponent("http://eb2.3lift.com/cset?key=uu&value=$UID");
                  var adnxsPart = "//ib.adnxs.com/getuid?"+base;
                  var i         =  document.createElement('IMG');
                  i.src         = adnxsPart;
                  i.display     = "none";
                  i.width       = 0;
                  i.height      = 0;
              });
          }
      };

      var cookie_domain = '.3lift.com';
      var cookieable = (function() {
          var hasCookie = docCookies.hasItem('tlcookieable');
          if (hasCookie) {
            checkAndSetUID();
            return true;
          }
          return false;
      })();
      console.log('Can set cookie', cookieable);

      var UserSync = function() {
          var protocol = window.location.protocol + '//';
          var num_user_sync_pixels = parseInt(getURLParameter('nusp'),10) || 0;
          var MS_BETWEEN_USER_SYNCS = 3 * 1000;
          var MS_WEEK = 1000 * 60 * 60 * 24 * 7;

          console.log('Sync pixels:',num_user_sync_pixels);

          var that = this;

          var getStartIdx = this.getStartIdx = function() {
            var sync_idx = parseInt(docCookies.getItem('tl_sync_idx'), 10);
            if (isNaN(sync_idx)) {
              sync_idx = -1;
            }
            var week_start = parseInt(docCookies.getItem('tl_sync_week_start'), 10) || 0;
            var now = (new Date()).getTime();

            console.log(sync_idx, week_start, now);

            // Start the sync if either idx or date is missing
            if ((sync_idx === -1) || !week_start) {
              docCookies.setItem('tl_sync_week_start', (new Date()).getTime(), Infinity, null, cookie_domain);
              return 0;
            }

            // Resume if there's more to do
            if ((sync_idx + 1) < num_user_sync_pixels) {
              return sync_idx + 1;
            }

            // Reset if we're over the # of pixels and it's been longer than a week
            if (((sync_idx + 1) >= num_user_sync_pixels) && ((week_start + MS_WEEK) <= now)) {
              docCookies.setItem('tl_sync_week_start', (new Date()).getTime(), Infinity, null, cookie_domain);
              return 0;
            }

            console.log('Reached end');

            return -1;
          };

          var makeUserSyncCalls = this.makeUserSyncCalls = function() {
            var start_idx = this.getStartIdx();

            console.log('Start idx:', start_idx);

            if (start_idx >= 0) {
              for (var i = start_idx; i < num_user_sync_pixels; i++) {
                console.log('Processing ' + i);
                setTimeout((function (i) {
                  return function() {
                    that.makeUserSyncCall(i);
                  };
                })(i), (i - start_idx) * MS_BETWEEN_USER_SYNCS); 
              }
            }
          };

          var makeUserSyncCall = this.makeUserSyncCall = function(idx) {
            var sync_url = createUrl(protocol + 'eb2.3lift.com/uidps', {'idx' : idx});
            var pix = document.createElement('IMG');
            pix.src = sync_url;
            pix.width = 0;
            pix.height = 0;
            document.body.appendChild(pix);
            removePixel(pix);
            docCookies.setItem('tl_sync_idx', idx, Infinity, null, cookie_domain);
          }
      };

      if (cookieable) {
        var userSync = new UserSync();
        userSync.makeUserSyncCalls();
      } else {
        console.log('Not cookieable');
      }
    