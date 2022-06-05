
            /*<![CDATA[*/
            if (window.bbcdotcom === undefined) {
                var bbcdotcom = {
                    data: {},
                    advert: {
                        write: function () {},
                        show: function () {},
                        isActive: function () {
                            return false;
                        }
                    },
                    objects: function(str) {
                        return false;
                    }
                };
            }
            bbcdotcom.objects = function (strName, action, obj) {
                var i,
                    len,
                    args = arguments,
                    nameParts = strName.split(".");
                if (action === undefined) {
                    action = 'valid';
                }
                if (obj === undefined) {
                    obj = window;
                }
                for (i = 0, len = nameParts.length; i < len; i++) {
                    if (obj[nameParts[i]] === undefined) {
                        if (action === 'create') {
                            obj[nameParts[i]] = {};
                        } else {
                            return false;
                        }
                    } else if (typeof obj[nameParts[i]] === 'function') {
                        if (typeof args[1] !== 'undefined' && args[1] !== null && obj[nameParts[i]](args[1]) !== undefined) {
                            return obj[nameParts[i]](args[1]);
                        } else if (obj[nameParts[i]]() !== undefined) {
                            return obj[nameParts[i]]();
                        }
                    }
                    obj = obj[nameParts[i]];
                }
                return obj;
            };
            /*]]>*/
        