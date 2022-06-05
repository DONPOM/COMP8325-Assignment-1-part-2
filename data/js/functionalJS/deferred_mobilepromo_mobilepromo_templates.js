(function() {
    var e=!1, t=!0, n = null, r = "checkbox", i = $.fn.extendCtx, s = $.fn.View, o = $.views, u = $.fn.renderContent;
    window.templates = {
        "#buttonTemplate": {
            markup: '     {{if notSubmitBtn}}     <a href="{{:url}}"         id="{{:identifier}}"         class="{{:classString}}"         onclick="{{:onClick}}"         data-original-title="{{>tooltip}}"         {{:extraAttributes}}         >         {{if inButtonGroup}}             <span>                 {{for #data tmpl="#buttonLabelTemplate" /}}             </span>         {{else}}             {{for #data tmpl="#buttonLabelTemplate" /}}         {{/if}}         </a>     {{else}}         <input type="submit" value="{{:label}}"         id="{{:identifier}}"         class="{{:classString}}"         onclick="{{:onClick}}"          {{:extraAttributes}}         >     {{/if}} ',
            tmpls: [{
                markup: '     <a href=\\"{{:url}}\\"         id=\\"{{:identifier}}\\"         class=\\"{{:classString}}\\"         onclick=\\"{{:onClick}}\\"         data-original-title=\\"{{>tooltip}}\\"         {{:extraAttributes}}         >         {{if inButtonGroup}}             <span>                 {{for #data tmpl=\\"#buttonLabelTemplate\\" /}}             </span>         {{else}}             {{for #data tmpl=\\"#buttonLabelTemplate\\" /}}         {{/if}}         </a>     ',
                tmpls: [{
                    markup: '             <span>                 {{for #data tmpl=\\"#buttonLabelTemplate\\" /}}             </span>         ',
                    tmpls: [],
                    links: {},
                    bnds: [],
                    render: function(t, n, r, i, s, o) {
                        return u.call(this, t, n, r, i, s, o)
                    },
                    htmlTag: "span",
                    fn: function(t, n, r, i) {
                        var r = r || jQuery.views, s = r._tag, o = "";
                        try {
                            o += "             <span>                 ";
                            o += s("for", n, this, "", {
                                props: {
                                    tmpl: "#buttonLabelTemplate"
                                },
                                params: '#data tmpl="#buttonLabelTemplate" '
                            }, 0, t);
                            o += "             </span>         ";
                            return o
                        } catch (u) {
                            return r._err(u)
                        }
                    }
                }, {
                    markup: '             {{for #data tmpl=\\"#buttonLabelTemplate\\" /}}         ',
                    tmpls: [],
                    links: {},
                    bnds: [],
                    render: function(t, n, r, i, s, o) {
                        return u.call(this, t, n, r, i, s, o)
                    },
                    htmlTag: "",
                    fn: function(t, n, r, i) {
                        var r = r || jQuery.views, s = r._tag, o = "";
                        try {
                            o += "             ";
                            o += s("for", n, this, "", {
                                props: {
                                    tmpl: "#buttonLabelTemplate"
                                },
                                params: '#data tmpl="#buttonLabelTemplate" '
                            }, 0, t);
                            o += "         ";
                            return o
                        } catch (u) {
                            return r._err(u)
                        }
                    }
                }
                ],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "a",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = r._tag, u = r.converters.html, a = "";
                    try {
                        a += '     <a href="';
                        a += (s = t.url) != i ? s : "";
                        a += '"         id="';
                        a += (s = t.identifier) != i ? s : "";
                        a += '"         class="';
                        a += (s = t.classString) != i ? s : "";
                        a += '"         onclick="';
                        a += (s = t.onClick) != i ? s : "";
                        a += '"         data-original-title="';
                        a += u(t.tooltip);
                        a += '"         ';
                        a += (s = t.extraAttributes) != i ? s : "";
                        a += "         >         ";
                        a += o("if", n, this, 1, {
                            params: "inButtonGroup"
                        }, 0, t.inButtonGroup);
                        a += o("else", n, this, 2, {
                            params: ""
                        }, 0);
                        a += "         </a>     ";
                        return a
                    } catch (f) {
                        return r._err(f)
                    }
                }
            }, {
                markup: '         <input type=\\"submit\\" value=\\"{{:label}}\\"         id=\\"{{:identifier}}\\"         class=\\"{{:classString}}\\"         onclick=\\"{{:onClick}}\\"          {{:extraAttributes}}         >     ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "input",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = "";
                    try {
                        o += '         <input type="submit" value="';
                        o += (s = t.label) != i ? s : "";
                        o += '"         id="';
                        o += (s = t.identifier) != i ? s : "";
                        o += '"         class="';
                        o += (s = t.classString) != i ? s : "";
                        o += '"         onclick="';
                        o += (s = t.onClick) != i ? s : "";
                        o += '"          ';
                        o += (s = t.extraAttributes) != i ? s : "";
                        o += "         >     ";
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "buttonTemplate",
            htmlTag: "a",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += "     ";
                    o += s("if", n, this, 1, {
                        params: "notSubmitBtn"
                    }, 0, t.notSubmitBtn);
                    o += s("else", n, this, 2, {
                        params: ""
                    }, 0);
                    o += " ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#buttonLabelTemplate": {
            markup: "     {{if !displayIconAfterLabel}}         <i class='{{:iconClass}}'></i>     {{/if}}     {{if label}}         {{>label}}     {{/if}}     {{if displayIconAfterLabel}}         <i class='{{:iconClass}}'></i>     {{/if}} ",
            tmpls: [{
                markup: "         <i class=\\'{{:iconClass}}\\'></i>     ",
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "i",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = "";
                    try {
                        o += "         <i class='";
                        o += (s = t.iconClass) != i ? s : "";
                        o += "'></i>     ";
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }, {
                markup: "         {{>label}}     ",
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = r.converters.html, o = "";
                    try {
                        o += "         ";
                        o += s(t.label);
                        o += "     ";
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }, {
                markup: "         <i class=\\'{{:iconClass}}\\'></i>     ",
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "i",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = "";
                    try {
                        o += "         <i class='";
                        o += (s = t.iconClass) != i ? s : "";
                        o += "'></i>     ";
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "buttonLabelTemplate",
            htmlTag: "i",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += "     ";
                    o += s("if", n, this, 1, {
                        params: "!displayIconAfterLabel"
                    }, 0, !t.displayIconAfterLabel);
                    o += "     ";
                    o += s("if", n, this, 2, {
                        params: "label"
                    }, 0, t.label);
                    o += "     ";
                    o += s("if", n, this, 3, {
                        params: "displayIconAfterLabel"
                    }, 0, t.displayIconAfterLabel);
                    o += " ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#stopwatchesTemplate": {
            markup: '<tr>         <th colspan="2">{{:url}}</td>     </tr>     <tr>         <td>{{:stopwatch.name}} ({{:~round(stopwatch.elapsed)}} s)</td>         <td width="1024px"><div class=\'time-bar\' style="width: 1024px;">&nbsp;</div></td>     </tr>     {{for stopwatch.children tmpl="#stopwatchesRowTemplate" /}}',
            tmpls: [],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "stopwatchesTemplate",
            htmlTag: "tr",
            _elCnt: !0,
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s, o = r._tag, u = "";
                try {
                    u += '     <tr>         <th colspan="2">';
                    u += (s = t.url) != i ? s : "";
                    u += "</td>     </tr>     <tr>         <td>";
                    u += (s = t.stopwatch.name) != i ? s : "";
                    u += " (";
                    u += (s = n._hlp("round")(t.stopwatch.elapsed)) != i ? s : "";
                    u += ' s)</td>         <td width="1024px"><div class=\'time-bar\' style="width: 1024px;">&nbsp;</div></td>     </tr>     ';
                    u += o("for", n, this, "", {
                        props: {
                            tmpl: "#stopwatchesRowTemplate"
                        },
                        params: 'stopwatch.children tmpl="#stopwatchesRowTemplate" '
                    }, 0, t.stopwatch.children);
                    u += " ";
                    return u
                } catch (a) {
                    return r._err(a)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#stopwatchesRowTemplate": {
            markup: '<tr>         <td width="300px">             {{if marker}}             <a href="javascript:void(0);" style=\'color: green;\' onclick="$(this).siblings(\'div\').toggle();">{{:~indent(#data)}} {{:~truncate(name)}} (at {{:~marker(#data)}} s)</a>             {{else}}             <a href="javascript:void(0);" onclick="$(this).siblings(\'div\').toggle();">{{:~indent(#data)}} {{:~truncate(name)}} ({{:~round(elapsed)}} s)</a>             {{/if}}             <div style="display: none;">                 <div><strong>{{:name}}</strong></div>                 <div>                     trace:</br>                     <div class=\'debug-pre\'>{{:trace}}</div>                 </div>             </div>         </td>         <td width="1024px">             {{if marker}}             <div class=\'time-bar\' style=" left: {{:~barLeft(#data)}}px; width: {{:~barWidth(#data)}}px; background-color: green;">&nbsp;</div>             {{else}}             <div class=\'time-bar\' style=" left: {{:~barLeft(#data)}}px; width: {{:~barWidth(#data)}}px;">&nbsp;</div>             {{/if}}         </td>     </tr>     {{for children tmpl="#stopwatchesRowTemplate" /}}',
            tmpls: [{
                markup: "             <a href=\\\"javascript:void(0);\\\" style=\\'color: green;\\' onclick=\\\"$(this).siblings(\\'div\\').toggle();\\\">{{:~indent(#data)}} {{:~truncate(name)}} (at {{:~marker(#data)}} s)</a>             ",
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "a",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = "";
                    try {
                        o += "             <a href=\"javascript:void(0);\" style='color: green;' onclick=\"$(this).siblings('div').toggle();\">";
                        o += (s = n._hlp("indent")(t)) != i ? s : "";
                        o += " ";
                        o += (s = n._hlp("truncate")(t.name)) != i ? s : "";
                        o += " (at ";
                        o += (s = n._hlp("marker")(t)) != i ? s : "";
                        o += " s)</a>             ";
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }, {
                markup: '             <a href=\\"javascript:void(0);\\" onclick=\\"$(this).siblings(\\\'div\\\').toggle();\\">{{:~indent(#data)}} {{:~truncate(name)}} ({{:~round(elapsed)}} s)</a>             ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "a",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = "";
                    try {
                        o += '             <a href="javascript:void(0);" onclick="$(this).siblings(\'div\').toggle();">';
                        o += (s = n._hlp("indent")(t)) != i ? s : "";
                        o += " ";
                        o += (s = n._hlp("truncate")(t.name)) != i ? s : "";
                        o += " (";
                        o += (s = n._hlp("round")(t.elapsed)) != i ? s : "";
                        o += " s)</a>             ";
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }, {
                markup: "             <div class=\\'time-bar\\' style=\\\" left: {{:~barLeft(#data)}}px; width: {{:~barWidth(#data)}}px; background-color: green;\\\">&nbsp;</div>             ",
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "div",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = "";
                    try {
                        o += "             <div class='time-bar' style=\" left: ";
                        o += (s = n._hlp("barLeft")(t)) != i ? s : "";
                        o += "px; width: ";
                        o += (s = n._hlp("barWidth")(t)) != i ? s : "";
                        o += 'px; background-color: green;">&nbsp;</div>             ';
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }, {
                markup: "             <div class=\\'time-bar\\' style=\\\" left: {{:~barLeft(#data)}}px; width: {{:~barWidth(#data)}}px;\\\">&nbsp;</div>             ",
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "div",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = "";
                    try {
                        o += "             <div class='time-bar' style=\" left: ";
                        o += (s = n._hlp("barLeft")(t)) != i ? s : "";
                        o += "px; width: ";
                        o += (s = n._hlp("barWidth")(t)) != i ? s : "";
                        o += 'px;">&nbsp;</div>             ';
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "stopwatchesRowTemplate",
            htmlTag: "tr",
            _elCnt: !0,
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s, o = r._tag, u = "";
                try {
                    u += '     <tr>         <td width="300px">             ';
                    u += o("if", n, this, 1, {
                        params: "marker"
                    }, 0, t.marker);
                    u += o("else", n, this, 2, {
                        params: ""
                    }, 0);
                    u += '             <div style="display: none;">                 <div><strong>';
                    u += (s = t.name) != i ? s : "";
                    u += "</strong></div>                 <div>                     trace:</br>                     <div class='debug-pre'>";
                    u += (s = t.trace) != i ? s : "";
                    u += '</div>                 </div>             </div>         </td>         <td width="1024px">             ';
                    u += o("if", n, this, 3, {
                        params: "marker"
                    }, 0, t.marker);
                    u += o("else", n, this, 4, {
                        params: ""
                    }, 0);
                    u += "         </td>     </tr>     ";
                    u += o("for", n, this, "", {
                        props: {
                            tmpl: "#stopwatchesRowTemplate"
                        },
                        params: 'children tmpl="#stopwatchesRowTemplate" '
                    }, 0, t.children);
                    u += " ";
                    return u
                } catch (a) {
                    return r._err(a)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#debugMessagesTemplate": {
            markup: "<tr>         <th>{{:url}}</th>     </tr>     {{for messages }}     <tr>         <td class='debug-pre {{:~messageStyle(#data)}}'>{{:#data}}</td>     </tr>     {{/for}}",
            tmpls: [{
                markup: "<tr>         <td class=\\'debug-pre {{:~messageStyle(#data)}}\\'>{{:#data}}</td>     </tr>",
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "tr",
                _elCnt: !0,
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = "";
                    try {
                        o += "     <tr>         <td class='debug-pre ";
                        o += (s = n._hlp("messageStyle")(t)) != i ? s : "";
                        o += "'>";
                        o += (s = t) != i ? s : "";
                        o += "</td>     </tr>     ";
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "debugMessagesTemplate",
            htmlTag: "tr",
            _elCnt: !0,
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s, o = r._tag, u = "";
                try {
                    u += "     <tr>         <th>";
                    u += (s = t.url) != i ? s : "";
                    u += "</th>     </tr>     ";
                    u += o("for", n, this, 1, {
                        params: "messages "
                    }, 0, t.messages);
                    u += " ";
                    return u
                } catch (a) {
                    return r._err(a)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#profilePicTemplate": {
            markup: " <img data-link=\"src{:url} data-size{:sizeClass} class{:'avatar' + (isViewerPic ? ' ownerAvatar' : '')}\" alt=\"\" /> ",
            tmpls: [],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "profilePicTemplate",
            htmlTag: "img",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = "";
                try {
                    s += " <img data-link=\"src{:url} data-size{:sizeClass} class{:'avatar' + (isViewerPic ? ' ownerAvatar' : '')}\" alt=\"\" /> ";
                    return s
                } catch (o) {
                    return r._err(o)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#loginBoxTemplate": {
            markup: '     <div id=\'loginBox\'>         <div id="login-left">             {{for facebookLogin tmpl="#externalSiteRegisterTemplate" /}}             {{for twitterLogin tmpl="#externalSiteRegisterTemplate" /}}             <p class="signup">                 Need an account?<br/>                 <a data-link="href{:registerUrl}">Sign up!</a>             </p>         </div>         <div id="login-divider"></div>         <div id="login-right">             <iframe id=\'loginBoxFrame\' name=\'loginBoxFrame\' data-link="src{:iframeUrl}" frameborder="0"></iframe>         </div>         <div class="clearB"></div>     </div> ',
            tmpls: [],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "loginBoxTemplate",
            htmlTag: "div",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += "     <div id='loginBox'>         <div id=\"login-left\">             ";
                    o += s("for", n, this, "", {
                        props: {
                            tmpl: "#externalSiteRegisterTemplate"
                        },
                        params: 'facebookLogin tmpl="#externalSiteRegisterTemplate" '
                    }, 0, t.facebookLogin);
                    o += "             ";
                    o += s("for", n, this, "", {
                        props: {
                            tmpl: "#externalSiteRegisterTemplate"
                        },
                        params: 'twitterLogin tmpl="#externalSiteRegisterTemplate" '
                    }, 0, t.twitterLogin);
                    o += '             <p class="signup">                 Need an account?<br/>                 <a data-link="href{:registerUrl}">Sign up!</a>             </p>         </div>         <div id="login-divider"></div>         <div id="login-right">             <iframe id=\'loginBoxFrame\' name=\'loginBoxFrame\' data-link="src{:iframeUrl}" frameborder="0"></iframe>         </div>         <div class="clearB"></div>     </div> ';
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#externalSiteRegisterTemplate": {
            markup: "     <a data-link=\"class{:'register-external register-'+service+' bigWeb'} href{:registerUrl} onclick{:~getExternalSiteRegisterClickHandler(tracking)}\">         <div id={{:'register-'+service+'-button'}} class=\"register-button\">             <div class=\"register-icon\">                 <i class={{:'icon-'+service}}></i>             </div>             <div class=\"register-text\">                 {{>buttonText}}             </div>         </div>     </a> ",
            tmpls: [],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "externalSiteRegisterTemplate",
            htmlTag: "a",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s, o = r.converters.html, u = "";
                try {
                    u += "     <a data-link=\"class{:'register-external register-'+service+' bigWeb'} href{:registerUrl} onclick{:~getExternalSiteRegisterClickHandler(tracking)}\">         <div id=";
                    u += (s = "register-" + t.service + "-button") != i ? s : "";
                    u += ' class="register-button">             <div class="register-icon">                 <i class=';
                    u += (s = "icon-" + t.service) != i ? s : "";
                    u += '></i>             </div>             <div class="register-text">                 ';
                    u += o(t.buttonText);
                    u += "             </div>         </div>     </a> ";
                    return u
                } catch (a) {
                    return r._err(a)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#editCreateAlbumModalTemplate": {
            markup: '      <input type="hidden" value="" name="parentAlbumPath" />     <input type="hidden" value="" name="albumPath" />     <div class="settingsNotice">         Updating the following settings will only affect the selected album and <strong>will not</strong> affect any other albums or sub-albums.     </div>      <div class="editAlbumTitle editalbum-section">         <div class="editAlbumSectionContent">             <div class="editAlbumSectionTitle titleTitle nameError">Album Name</div>             {{for albumTitle tmpl="#inputTextTemplate" /}}             <p class="warning">Warning: Renaming the album will break your direct links, slideshows, and stories!</p>         </div>     </div>      {{if isHashtagPartner}}         <div class="createHashtagSectionContent editalbum-section">             <div class="createHashtagSectionTitle">Create a hashtag for your new album</div>             <img class="hashtag" src="{{:pbHashtagUrl}}">             {{for createHashtag tmpl="#inputTextTemplate" /}}         </div>         <div class=editHashtagSectionContent editAlbum-section">             {{if hashtag }}                 <div class="editHashtagSectionTitle">Hashtag for this album</div>                 <div class="editHashtagContent"><div class="hashtagLabel">{{:hashtag.hashtag}}</div> {{for toggleHashtag tmpl="#toggleTemplate" /}} </div>                 <div class="deleteHashtagSectionTitle">Would you like to delete this hashtag?</div>                 <input type="hidden" name="hashtagId" value="{{:hashtag.id}}"/>                 <div class="deleteHashtagContent"> {{for deleteHashtag tmpl="#buttonTemplate" /}} </div>             {{else}}                 <div class="createHashtagSectionTitle">Add a hashtag to this album</div>                 <img class="hashtag" src="{{:pbHashtagUrl}}">                 {{for createHashtag tmpl="#inputTextTemplate" /}}             {{/if}}         </div>     {{/if}}     <div class="editAlbumDescription editalbum-section">         <div class="editAlbumSectionContent">             <div class="editAlbumSectionTitle">Description <span class="help-inline"></span></div>             <div class="descContainer" id="descContainer">                 <div class="descInputContainer">                 {{for albumDescription tmpl="#textareaTemplate" /}}                 </div>             </div>         </div>     </div>     <div class="editAlbumPrivacy editalbum-section">         <i class="icon-lock"></i>         <div class="editAlbumSectionContent">             <div class="privacyOptionsCollapsed privacyTitle" id="privacyOptionsCollapsed">                 <span class="editAlbumSectionTitle">Privacy Setting: </span>                 <span class="privacyTitleCurrentSetting" id="privacyTitleCurrentSetting">Public</span>&nbsp;                 <strong>(<a href="#" class="privacyEditButton" id="privacyEditButton">edit</a>)</strong>             </div>             <div class="privacyOptions hide" id="privacyOptions">                 <div class="editAlbumSectionTitle privacyTitle">Privacy Settings:</div>                 <div class="privacySectionsContainer">                     <div class="pull-left privacyLeftSection">                         <div>                             <div class="radio"><input type="radio" name="privacyLevel" value="public" id="privacyLevelPublic" /></div>                             <div class="radioContent">                                 <div class="radioContentTitle">                                     Public                                 </div>                                 <div class="radioContentDescription">                                     Everyone can access this album.                                 </div>                             </div>                         </div>                     </div>                     <div class="pull-left privacyRightSection">                         <div class="radio">                             <input type="radio" name="privacyLevel" value="private" id="privacyLevelPrivate" />                         </div>                         <div class="radioContent">                             <div class="radioContentTitle">                                 Private                             </div>                             <div class="radioContentDescription">                                 <div>Only you can see this album. You can share this later with specific people or groups.</div>                                 <br>                                 <div><input class="checkbox" type="checkbox" disabled name="passwordProtect" id="privacyPasswordCheckbox" autocomplete="off"/>Guest password<a id="showOrHide" href="#">Show</a></div>                                 <div class="passwordContainer">                                     <input class="password" type="password" disabled name="privacyPassword" id="privacyPassword" autocomplete="off">                                     <div class="help-inline">Invalid password (Passwords must be at least 6 characters long. They can only contain the characters a-z, A-Z, 0-9, -, and _. Spaces are not allowed.)</div>                                 </div>                             </div>                         </div>                     </div>                     <div class="clearB"></div>                 </div>             </div>         </div>     </div>      <div class="editalbum-section">         <div class="editAlbumSectionContent editAlbumSelector">             <div class="editAlbumSectionTitle editAlbumSelectorText parentAlbumPathError">Add this as a sub-album to: </div>             <div class="help-block"></div>             {{for chooser tmpl="#albumChooserDropdownTemplate" /}}         </div>     </div> ',
            tmpls: [{
                markup: '         <div class=\\"createHashtagSectionContent editalbum-section\\">             <div class=\\"createHashtagSectionTitle\\">Create a hashtag for your new album</div>             <img class=\\"hashtag\\" src=\\"{{:pbHashtagUrl}}\\">             {{for createHashtag tmpl=\\"#inputTextTemplate\\" /}}         </div>         <div class=editHashtagSectionContent editAlbum-section\\">             {{if hashtag }}                 <div class=\\"editHashtagSectionTitle\\">Hashtag for this album</div>                 <div class=\\"editHashtagContent\\"><div class=\\"hashtagLabel\\">{{:hashtag.hashtag}}</div> {{for toggleHashtag tmpl=\\"#toggleTemplate\\" /}} </div>                 <div class=\\"deleteHashtagSectionTitle\\">Would you like to delete this hashtag?</div>                 <input type=\\"hidden\\" name=\\"hashtagId\\" value=\\"{{:hashtag.id}}\\"/>                 <div class=\\"deleteHashtagContent\\"> {{for deleteHashtag tmpl=\\"#buttonTemplate\\" /}} </div>             {{else}}                 <div class=\\"createHashtagSectionTitle\\">Add a hashtag to this album</div>                 <img class=\\"hashtag\\" src=\\"{{:pbHashtagUrl}}\\">                 {{for createHashtag tmpl=\\"#inputTextTemplate\\" /}}             {{/if}}         </div>     ',
                tmpls: [{
                    markup: '                 <div class=\\"editHashtagSectionTitle\\">Hashtag for this album</div>                 <div class=\\"editHashtagContent\\"><div class=\\"hashtagLabel\\">{{:hashtag.hashtag}}</div> {{for toggleHashtag tmpl=\\"#toggleTemplate\\" /}} </div>                 <div class=\\"deleteHashtagSectionTitle\\">Would you like to delete this hashtag?</div>                 <input type=\\"hidden\\" name=\\"hashtagId\\" value=\\"{{:hashtag.id}}\\"/>                 <div class=\\"deleteHashtagContent\\"> {{for deleteHashtag tmpl=\\"#buttonTemplate\\" /}} </div>             ',
                    tmpls: [],
                    links: {},
                    bnds: [],
                    render: function(t, n, r, i, s, o) {
                        return u.call(this, t, n, r, i, s, o)
                    },
                    htmlTag: "div",
                    fn: function(t, n, r, i) {
                        var r = r || jQuery.views, s, o = r._tag, u = "";
                        try {
                            u += '                 <div class="editHashtagSectionTitle">Hashtag for this album</div>                 <div class="editHashtagContent"><div class="hashtagLabel">';
                            u += (s = t.hashtag.hashtag) != i ? s : "";
                            u += "</div> ";
                            u += o("for", n, this, "", {
                                props: {
                                    tmpl: "#toggleTemplate"
                                },
                                params: 'toggleHashtag tmpl="#toggleTemplate" '
                            }, 0, t.toggleHashtag);
                            u += ' </div>                 <div class="deleteHashtagSectionTitle">Would you like to delete this hashtag?</div>                 <input type="hidden" name="hashtagId" value="';
                            u += (s = t.hashtag.id) != i ? s : "";
                            u += '"/>                 <div class="deleteHashtagContent"> ';
                            u += o("for", n, this, "", {
                                props: {
                                    tmpl: "#buttonTemplate"
                                },
                                params: 'deleteHashtag tmpl="#buttonTemplate" '
                            }, 0, t.deleteHashtag);
                            u += " </div>             ";
                            return u
                        } catch (a) {
                            return r._err(a)
                        }
                    }
                }, {
                    markup: '                 <div class=\\"createHashtagSectionTitle\\">Add a hashtag to this album</div>                 <img class=\\"hashtag\\" src=\\"{{:pbHashtagUrl}}\\">                 {{for createHashtag tmpl=\\"#inputTextTemplate\\" /}}             ',
                    tmpls: [],
                    links: {},
                    bnds: [],
                    render: function(t, n, r, i, s, o) {
                        return u.call(this, t, n, r, i, s, o)
                    },
                    htmlTag: "div",
                    fn: function(t, n, r, i) {
                        var r = r || jQuery.views, s, o = r._tag, u = "";
                        try {
                            u += '                 <div class="createHashtagSectionTitle">Add a hashtag to this album</div>                 <img class="hashtag" src="';
                            u += (s = t.pbHashtagUrl) != i ? s : "";
                            u += '">                 ';
                            u += o("for", n, this, "", {
                                props: {
                                    tmpl: "#inputTextTemplate"
                                },
                                params: 'createHashtag tmpl="#inputTextTemplate" '
                            }, 0, t.createHashtag);
                            u += "             ";
                            return u
                        } catch (a) {
                            return r._err(a)
                        }
                    }
                }
                ],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "div",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = r._tag, u = "";
                    try {
                        u += '         <div class="createHashtagSectionContent editalbum-section">             <div class="createHashtagSectionTitle">Create a hashtag for your new album</div>             <img class="hashtag" src="';
                        u += (s = t.pbHashtagUrl) != i ? s : "";
                        u += '">             ';
                        u += o("for", n, this, "", {
                            props: {
                                tmpl: "#inputTextTemplate"
                            },
                            params: 'createHashtag tmpl="#inputTextTemplate" '
                        }, 0, t.createHashtag);
                        u += '         </div>         <div class=editHashtagSectionContent editAlbum-section">             ';
                        u += o("if", n, this, 1, {
                            params: "hashtag "
                        }, 0, t.hashtag);
                        u += o("else", n, this, 2, {
                            params: ""
                        }, 0);
                        u += "         </div>     ";
                        return u
                    } catch (a) {
                        return r._err(a)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "editCreateAlbumModalTemplate",
            htmlTag: "input",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += '      <input type="hidden" value="" name="parentAlbumPath" />     <input type="hidden" value="" name="albumPath" />     <div class="settingsNotice">         Updating the following settings will only affect the selected album and <strong>will not</strong> affect any other albums or sub-albums.     </div>      <div class="editAlbumTitle editalbum-section">         <div class="editAlbumSectionContent">             <div class="editAlbumSectionTitle titleTitle nameError">Album Name</div>             ';
                    o += s("for", n, this, "", {
                        props: {
                            tmpl: "#inputTextTemplate"
                        },
                        params: 'albumTitle tmpl="#inputTextTemplate" '
                    }, 0, t.albumTitle);
                    o += '             <p class="warning">Warning: Renaming the album will break your direct links, slideshows, and stories!</p>         </div>     </div>      ';
                    o += s("if", n, this, 1, {
                        params: "isHashtagPartner"
                    }, 0, t.isHashtagPartner);
                    o += '     <div class="editAlbumDescription editalbum-section">         <div class="editAlbumSectionContent">             <div class="editAlbumSectionTitle">Description <span class="help-inline"></span></div>             <div class="descContainer" id="descContainer">                 <div class="descInputContainer">                 ';
                    o += s("for", n, this, "", {
                        props: {
                            tmpl: "#textareaTemplate"
                        },
                        params: 'albumDescription tmpl="#textareaTemplate" '
                    }, 0, t.albumDescription);
                    o += '                 </div>             </div>         </div>     </div>     <div class="editAlbumPrivacy editalbum-section">         <i class="icon-lock"></i>         <div class="editAlbumSectionContent">             <div class="privacyOptionsCollapsed privacyTitle" id="privacyOptionsCollapsed">                 <span class="editAlbumSectionTitle">Privacy Setting: </span>                 <span class="privacyTitleCurrentSetting" id="privacyTitleCurrentSetting">Public</span>&nbsp;                 <strong>(<a href="#" class="privacyEditButton" id="privacyEditButton">edit</a>)</strong>             </div>             <div class="privacyOptions hide" id="privacyOptions">                 <div class="editAlbumSectionTitle privacyTitle">Privacy Settings:</div>                 <div class="privacySectionsContainer">                     <div class="pull-left privacyLeftSection">                         <div>                             <div class="radio"><input type="radio" name="privacyLevel" value="public" id="privacyLevelPublic" /></div>                             <div class="radioContent">                                 <div class="radioContentTitle">                                     Public                                 </div>                                 <div class="radioContentDescription">                                     Everyone can access this album.                                 </div>                             </div>                         </div>                     </div>                     <div class="pull-left privacyRightSection">                         <div class="radio">                             <input type="radio" name="privacyLevel" value="private" id="privacyLevelPrivate" />                         </div>                         <div class="radioContent">                             <div class="radioContentTitle">                                 Private                             </div>                             <div class="radioContentDescription">                                 <div>Only you can see this album. You can share this later with specific people or groups.</div>                                 <br>                                 <div><input class="checkbox" type="checkbox" disabled name="passwordProtect" id="privacyPasswordCheckbox" autocomplete="off"/>Guest password<a id="showOrHide" href="#">Show</a></div>                                 <div class="passwordContainer">                                     <input class="password" type="password" disabled name="privacyPassword" id="privacyPassword" autocomplete="off">                                     <div class="help-inline">Invalid password (Passwords must be at least 6 characters long. They can only contain the characters a-z, A-Z, 0-9, -, and _. Spaces are not allowed.)</div>                                 </div>                             </div>                         </div>                     </div>                     <div class="clearB"></div>                 </div>             </div>         </div>     </div>      <div class="editalbum-section">         <div class="editAlbumSectionContent editAlbumSelector">             <div class="editAlbumSectionTitle editAlbumSelectorText parentAlbumPathError">Add this as a sub-album to: </div>             <div class="help-block"></div>             ';
                    o += s("for", n, this, "", {
                        props: {
                            tmpl: "#albumChooserDropdownTemplate"
                        },
                        params: 'chooser tmpl="#albumChooserDropdownTemplate" '
                    }, 0, t.chooser);
                    o += "         </div>     </div> ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#inputTextTemplate": {
            markup: '     <fieldset class="control-group">         {{if label}}             <label data-link="for{:name} {:label}" class="control-label"></label>          {{/if}}         <div class="controls">             <input data-link="type{:inputType} name{:name} value{:value} id{:id} class{:className} data-original-title{:tooltip} placeholder{:placeholderText ? placeholderText : \'\'}" {{:required}}>         </div>         {{if helpblock}}             <p class="help-block"></p>         {{else}}             <span class="help-inline"></span>         {{/if}}     </fieldset> ',
            tmpls: [{
                markup: '             <label data-link=\\"for{:name} {:label}\\" class=\\"control-label\\"></label>          ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "label",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '             <label data-link="for{:name} {:label}" class="control-label"></label>          ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }, {
                markup: '             <p class=\\"help-block\\"></p>         ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "p",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '             <p class="help-block"></p>         ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }, {
                markup: '             <span class=\\"help-inline\\"></span>         ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "span",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '             <span class="help-inline"></span>         ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "inputTextTemplate",
            htmlTag: "fieldset",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s, o = r._tag, u = "";
                try {
                    u += '     <fieldset class="control-group">         ';
                    u += o("if", n, this, 1, {
                        params: "label"
                    }, 0, t.label);
                    u += '         <div class="controls">             <input data-link="type{:inputType} name{:name} value{:value} id{:id} class{:className} data-original-title{:tooltip} placeholder{:placeholderText ? placeholderText : \'\'}" ';
                    u += (s = t.required) != i ? s : "";
                    u += ">         </div>         ";
                    u += o("if", n, this, 2, {
                        params: "helpblock"
                    }, 0, t.helpblock);
                    u += o("else", n, this, 3, {
                        params: ""
                    }, 0);
                    u += "     </fieldset> ";
                    return u
                } catch (a) {
                    return r._err(a)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#textareaTemplate": {
            markup: '     <fieldset class="control-group">         {{if label}}             <label data-link="for{:name} {:label}" class="control-label"></label>          {{/if}}         <div class="controls">             {{if iconClass}}                 <div data-link="class{:iconClass}"></div>             {{/if}}             <textarea data-link="name{:name} id{:id} data-original-title{:tooltip} placeholder{:placeholderText ? placeholderText : \'\'} {>value}"></textarea>         </div>         {{if helpblock}}             <p class="help-block"></p>         {{else}}             <span class="help-inline"></span>         {{/if}}     </fieldset> ',
            tmpls: [{
                markup: '             <label data-link=\\"for{:name} {:label}\\" class=\\"control-label\\"></label>          ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "label",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '             <label data-link="for{:name} {:label}" class="control-label"></label>          ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }, {
                markup: '                 <div data-link=\\"class{:iconClass}\\"></div>             ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "div",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '                 <div data-link="class{:iconClass}"></div>             ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }, {
                markup: '             <p class=\\"help-block\\"></p>         ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "p",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '             <p class="help-block"></p>         ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }, {
                markup: '             <span class=\\"help-inline\\"></span>         ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "span",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '             <span class="help-inline"></span>         ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "textareaTemplate",
            htmlTag: "fieldset",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += '     <fieldset class="control-group">         ';
                    o += s("if", n, this, 1, {
                        params: "label"
                    }, 0, t.label);
                    o += '         <div class="controls">             ';
                    o += s("if", n, this, 2, {
                        params: "iconClass"
                    }, 0, t.iconClass);
                    o += "             <textarea data-link=\"name{:name} id{:id} data-original-title{:tooltip} placeholder{:placeholderText ? placeholderText : ''} {>value}\"></textarea>         </div>         ";
                    o += s("if", n, this, 3, {
                        params: "helpblock"
                    }, 0, t.helpblock);
                    o += s("else", n, this, 4, {
                        params: ""
                    }, 0);
                    o += "     </fieldset> ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#albumChooserDropdownTemplate": {
            markup: '     <div class="albumChooser" data-link="id{:identifier}">         <span class="dropdown">             <a href="#" data-link="class{:(\'btn dropdown-toggle\' + buttonClass)}" data-toggle="dropdown"><span>Select Album</span> <b class="caret"></b></a>             <ul class="albumChooserList dropdown-menu hide albumchooser-container">             </ul>         </span>          <div class="progress progress-info progress-striped active hide">             <div class="bar" style="width: 100%;">Loading albums...</div>         </div>     </div> ',
            tmpls: [],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "albumChooserDropdownTemplate",
            htmlTag: "div",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = "";
                try {
                    s += '     <div class="albumChooser" data-link="id{:identifier}">         <span class="dropdown">             <a href="#" data-link="class{:(\'btn dropdown-toggle\' + buttonClass)}" data-toggle="dropdown"><span>Select Album</span> <b class="caret"></b></a>             <ul class="albumChooserList dropdown-menu hide albumchooser-container">             </ul>         </span>          <div class="progress progress-info progress-striped active hide">             <div class="bar" style="width: 100%;">Loading albums...</div>         </div>     </div> ';
                    return s
                } catch (o) {
                    return r._err(o)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#albumChooserDropdownItemTemplate": {
            markup: '  {{if divider}}     <li class="divider"></li> {{else}}     <li><a href="javascript:void(0);" data-link="{>displayName} class{:optionClass} data-albumpath{:path} data-albumshortpath{:shortPath} data-albumurl{:url} data-name{:name} data-privacy{:privacy}"></a></li> {{/if}}  ',
            tmpls: [{
                markup: '     <li class=\\"divider\\"></li> ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "li",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '     <li class="divider"></li> ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }, {
                markup: '     <li><a href=\\"javascript:void(0);\\" data-link=\\"{>displayName} class{:optionClass} data-albumpath{:path} data-albumshortpath{:shortPath} data-albumurl{:url} data-name{:name} data-privacy{:privacy}\\"></a></li> ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "li",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '     <li><a href="javascript:void(0);" data-link="{>displayName} class{:optionClass} data-albumpath{:path} data-albumshortpath{:shortPath} data-albumurl{:url} data-name{:name} data-privacy{:privacy}"></a></li> ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "albumChooserDropdownItemTemplate",
            htmlTag: "li",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += "  ";
                    o += s("if", n, this, 1, {
                        params: "divider"
                    }, 0, t.divider);
                    o += s("else", n, this, 2, {
                        params: ""
                    }, 0);
                    o += "  ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#toggleTemplate": {
            markup: '     <div class="toggleContainer" id="{{:id}}" data-name="{{:name}}">         <div class="toggle {{:on}}">             <span class="ballOn">On</span>             <div class="ball">|||</div>             <span class="ballOff">Off</span>         </div>     </div> ',
            tmpls: [],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "toggleTemplate",
            htmlTag: "div",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s, o = "";
                try {
                    o += '     <div class="toggleContainer" id="';
                    o += (s = t.id) != i ? s : "";
                    o += '" data-name="';
                    o += (s = t.name) != i ? s : "";
                    o += '">         <div class="toggle ';
                    o += (s = t.on) != i ? s : "";
                    o += '">             <span class="ballOn">On</span>             <div class="ball">|||</div>             <span class="ballOff">Off</span>         </div>     </div> ';
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#loginModalTemplate": {
            markup: '     {{for loginBox tmpl="#loginBoxTemplate" /}} ',
            tmpls: [],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "loginModalTemplate",
            htmlTag: "",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += "     ";
                    o += s("for", n, this, "", {
                        props: {
                            tmpl: "#loginBoxTemplate"
                        },
                        params: 'loginBox tmpl="#loginBoxTemplate" '
                    }, 0, t.loginBox);
                    o += " ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#modalTemplate": {
            markup: ' <div data-link="class{:className + \' modal hide fade\'} id{:id}" {{:static}} data-keyboard="{{:keyboard}}">     {{if includeForm}}         <form method="post">             {{for #data tmpl="#modalContentsTemplate" /}}         </form>     {{else}}         {{for #data tmpl="#modalContentsTemplate" /}}     {{/if}} </div> ',
            tmpls: [{
                markup: '         <form method=\\"post\\">             {{for #data tmpl=\\"#modalContentsTemplate\\" /}}         </form>     ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "form",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = r._tag, o = "";
                    try {
                        o += '         <form method="post">             ';
                        o += s("for", n, this, "", {
                            props: {
                                tmpl: "#modalContentsTemplate"
                            },
                            params: '#data tmpl="#modalContentsTemplate" '
                        }, 0, t);
                        o += "         </form>     ";
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }, {
                markup: '         {{for #data tmpl=\\"#modalContentsTemplate\\" /}}     ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = r._tag, o = "";
                    try {
                        o += "         ";
                        o += s("for", n, this, "", {
                            props: {
                                tmpl: "#modalContentsTemplate"
                            },
                            params: '#data tmpl="#modalContentsTemplate" '
                        }, 0, t);
                        o += "     ";
                        return o
                    } catch (u) {
                        return r._err(u)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "modalTemplate",
            htmlTag: "div",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s, o = r._tag, u = "";
                try {
                    u += " <div data-link=\"class{:className + ' modal hide fade'} id{:id}\" ";
                    u += (s = t.static) != i ? s : "";
                    u += ' data-keyboard="';
                    u += (s = t.keyboard) != i ? s : "";
                    u += '">     ';
                    u += o("if", n, this, 1, {
                        params: "includeForm"
                    }, 0, t.includeForm);
                    u += o("else", n, this, 2, {
                        params: ""
                    }, 0);
                    u += " </div> ";
                    return u
                } catch (a) {
                    return r._err(a)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#modalContentsTemplate": {
            markup: ' <div class="modal-header">     {{if !forceAction}}     <a class="close" data-dismiss="modal"><i class=\'icon-remove\'></i></a>     {{/if}}     <h2 class=\'title\' data-link="title"></h2> </div> <div class="modal-body">     {{for #data tmpl=templateName /}} </div> {{if hasFooter}}     <div class="modal-footer">         <span class="modal-footer-note">{{:footerNote}}</span>         {{for buttons tmpl="#buttonTemplate" /}}     </div> {{/if}} ',
            tmpls: [{
                markup: '     <a class=\\"close\\" data-dismiss=\\"modal\\"><i class=\\\'icon-remove\\\'></i></a>     ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "a",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = "";
                    try {
                        s += '     <a class="close" data-dismiss="modal"><i class=\'icon-remove\'></i></a>     ';
                        return s
                    } catch (o) {
                        return r._err(o)
                    }
                }
            }, {
                markup: '     <div class=\\"modal-footer\\">         <span class=\\"modal-footer-note\\">{{:footerNote}}</span>         {{for buttons tmpl=\\"#buttonTemplate\\" /}}     </div> ',
                tmpls: [],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "div",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = r._tag, u = "";
                    try {
                        u += '     <div class="modal-footer">         <span class="modal-footer-note">';
                        u += (s = t.footerNote) != i ? s : "";
                        u += "</span>         ";
                        u += o("for", n, this, "", {
                            props: {
                                tmpl: "#buttonTemplate"
                            },
                            params: 'buttons tmpl="#buttonTemplate" '
                        }, 0, t.buttons);
                        u += "     </div> ";
                        return u
                    } catch (a) {
                        return r._err(a)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "modalContentsTemplate",
            htmlTag: "div",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += ' <div class="modal-header">     ';
                    o += s("if", n, this, 1, {
                        params: "!forceAction"
                    }, 0, !t.forceAction);
                    o += '     <h2 class=\'title\' data-link="title"></h2> </div> <div class="modal-body">     ';
                    o += s("for", n, this, "", {
                        props: {
                            tmpl: t.templateName
                        },
                        params: "#data tmpl=templateName "
                    }, 0, t);
                    o += " </div> ";
                    o += s("if", n, this, 2, {
                        params: "hasFooter"
                    }, 0, t.hasFooter);
                    o += " ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#breadcrumbsItemTemplate": {
            markup: ' {{if ~isCurrentBreadcrumb(link)}}     <li>     	<span class=\'breadcrumbLabel\'>              {{:~truncate(label)}}         </span>         {{if privacy}}             {{if privacy === \'password\'}}                 <i class="icon-key"></i>             {{else privacy === \'public\'}}                 <i class="icon-unlock"></i>             {{else privacy === \'private\'}}                 <i class="icon-lock"></i>             {{/if}}         {{/if}}     </li> {{else}}     <li>     	<span class=\'breadcrumbLabel\'>             {{if root }}                  <a data-link="href{:profileUrl}">{{>profileLabel}}</a>             {{/if}}             <a data-link="href{:link}">{{>~truncate(label)}}</a>         </span>         <span class="divider">&#47;</span>     </li> {{/if}} ',
            tmpls: [{
                markup: "     <li>     	<span class=\\'breadcrumbLabel\\'>              {{:~truncate(label)}}         </span>         {{if privacy}}             {{if privacy === \\'password\\'}}                 <i class=\\\"icon-key\\\"></i>             {{else privacy === \\'public\\'}}                 <i class=\\\"icon-unlock\\\"></i>             {{else privacy === \\'private\\'}}                 <i class=\\\"icon-lock\\\"></i>             {{/if}}         {{/if}}     </li> ",
                tmpls: [{
                    markup: "             {{if privacy === \\'password\\'}}                 <i class=\\\"icon-key\\\"></i>             {{else privacy === \\'public\\'}}                 <i class=\\\"icon-unlock\\\"></i>             {{else privacy === \\'private\\'}}                 <i class=\\\"icon-lock\\\"></i>             {{/if}}         ",
                    tmpls: [{
                        markup: '                 <i class=\\"icon-key\\"></i>             ',
                        tmpls: [],
                        links: {},
                        bnds: [],
                        render: function(t, n, r, i, s, o) {
                            return u.call(this, t, n, r, i, s, o)
                        },
                        htmlTag: "i",
                        fn: function(t, n, r, i) {
                            var r = r || jQuery.views, s = "";
                            try {
                                s += '                 <i class="icon-key"></i>             ';
                                return s
                            } catch (o) {
                                return r._err(o)
                            }
                        }
                    }, {
                        markup: '                 <i class=\\"icon-unlock\\"></i>             ',
                        tmpls: [],
                        links: {},
                        bnds: [],
                        render: function(t, n, r, i, s, o) {
                            return u.call(this, t, n, r, i, s, o)
                        },
                        htmlTag: "i",
                        fn: function(t, n, r, i) {
                            var r = r || jQuery.views, s = "";
                            try {
                                s += '                 <i class="icon-unlock"></i>             ';
                                return s
                            } catch (o) {
                                return r._err(o)
                            }
                        }
                    }, {
                        markup: '                 <i class=\\"icon-lock\\"></i>             ',
                        tmpls: [],
                        links: {},
                        bnds: [],
                        render: function(t, n, r, i, s, o) {
                            return u.call(this, t, n, r, i, s, o)
                        },
                        htmlTag: "i",
                        fn: function(t, n, r, i) {
                            var r = r || jQuery.views, s = "";
                            try {
                                s += '                 <i class="icon-lock"></i>             ';
                                return s
                            } catch (o) {
                                return r._err(o)
                            }
                        }
                    }
                    ],
                    links: {},
                    bnds: [],
                    render: function(t, n, r, i, s, o) {
                        return u.call(this, t, n, r, i, s, o)
                    },
                    htmlTag: "i",
                    fn: function(t, n, r, i) {
                        var r = r || jQuery.views, s = r._tag, o = "";
                        try {
                            o += "             ";
                            o += s("if", n, this, 1, {
                                params: "privacy === 'password'"
                            }, 0, t.privacy === "password");
                            o += s("else", n, this, 2, {
                                params: "privacy === 'public'"
                            }, 0, t.privacy === "public");
                            o += s("else", n, this, 3, {
                                params: "privacy === 'private'"
                            }, 0, t.privacy === "private");
                            o += "         ";
                            return o
                        } catch (u) {
                            return r._err(u)
                        }
                    }
                }
                ],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "li",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s, o = r._tag, u = "";
                    try {
                        u += "     <li>     	<span class='breadcrumbLabel'>              ";
                        u += (s = n._hlp("truncate")(t.label)) != i ? s : "";
                        u += "         </span>         ";
                        u += o("if", n, this, 1, {
                            params: "privacy"
                        }, 0, t.privacy);
                        u += "     </li> ";
                        return u
                    } catch (a) {
                        return r._err(a)
                    }
                }
            }, {
                markup: '     <li>     	<span class=\\\'breadcrumbLabel\\\'>             {{if root }}                  <a data-link=\\"href{:profileUrl}\\">{{>profileLabel}}</a>             {{/if}}             <a data-link=\\"href{:link}\\">{{>~truncate(label)}}</a>         </span>         <span class=\\"divider\\">&#47;</span>     </li> ',
                tmpls: [{
                    markup: '                  <a data-link=\\"href{:profileUrl}\\">{{>profileLabel}}</a>             ',
                    tmpls: [],
                    links: {},
                    bnds: [],
                    render: function(t, n, r, i, s, o) {
                        return u.call(this, t, n, r, i, s, o)
                    },
                    htmlTag: "a",
                    fn: function(t, n, r, i) {
                        var r = r || jQuery.views, s = r.converters.html, o = "";
                        try {
                            o += '                  <a data-link="href{:profileUrl}">';
                            o += s(t.profileLabel);
                            o += "</a>             ";
                            return o
                        } catch (u) {
                            return r._err(u)
                        }
                    }
                }
                ],
                links: {},
                bnds: [],
                render: function(t, n, r, i, s, o) {
                    return u.call(this, t, n, r, i, s, o)
                },
                htmlTag: "li",
                fn: function(t, n, r, i) {
                    var r = r || jQuery.views, s = r._tag, o = r.converters.html, u = "";
                    try {
                        u += "     <li>     	<span class='breadcrumbLabel'>             ";
                        u += s("if", n, this, 1, {
                            params: "root "
                        }, 0, t.root);
                        u += '             <a data-link="href{:link}">';
                        u += o(n._hlp("truncate")(t.label));
                        u += '</a>         </span>         <span class="divider">&#47;</span>     </li> ';
                        return u
                    } catch (a) {
                        return r._err(a)
                    }
                }
            }
            ],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "breadcrumbsItemTemplate",
            htmlTag: "li",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += " ";
                    o += s("if", n, this, 1, {
                        params: "~isCurrentBreadcrumb(link)"
                    }, 0, n._hlp("isCurrentBreadcrumb")(t.link));
                    o += s("else", n, this, 2, {
                        params: ""
                    }, 0);
                    o += " ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#breadcrumbsTemplate": {
            markup: ' {{for breadcrumbs tmpl="#breadcrumbsItemTemplate" /}} ',
            tmpls: [],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "breadcrumbsTemplate",
            htmlTag: "",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s = r._tag, o = "";
                try {
                    o += " ";
                    o += s("for", n, this, "", {
                        props: {
                            tmpl: "#breadcrumbsItemTemplate"
                        },
                        params: 'breadcrumbs tmpl="#breadcrumbsItemTemplate" '
                    }, 0, t.breadcrumbs);
                    o += " ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        },
        "#alertTemplate": {
            markup: '     <div class="global alert fade slide         alert-{{:messageClass}}"         data-alert="alert">         <a class="close" data-dismiss="alert" href="#"><i class=\'icon-remove\'></i></a>         <div class="msg-body">             <i class="icon-{{:messageIcon}} icon-large"></i>             <div class="msg-title">{{:message}}</div>             <div class="msg-details">{{:details}}</div>         </div>     </div> ',
            tmpls: [],
            links: {},
            bnds: [],
            render: function(t, n, r, i, s, o) {
                return u.call(this, t, n, r, i, s, o)
            },
            tmplName: "alertTemplate",
            htmlTag: "div",
            fn: function(t, n, r, i) {
                var r = r || jQuery.views, s, o = "";
                try {
                    o += '     <div class="global alert fade slide         alert-';
                    o += (s = t.messageClass) != i ? s : "";
                    o += '"         data-alert="alert">         <a class="close" data-dismiss="alert" href="#"><i class=\'icon-remove\'></i></a>         <div class="msg-body">             <i class="icon-';
                    o += (s = t.messageIcon) != i ? s : "";
                    o += ' icon-large"></i>             <div class="msg-title">';
                    o += (s = t.message) != i ? s : "";
                    o += '</div>             <div class="msg-details">';
                    o += (s = t.details) != i ? s : "";
                    o += "</div>         </div>     </div> ";
                    return o
                } catch (u) {
                    return r._err(u)
                }
            },
            _is: "template",
            link: function(t, n, r, i, s, o, u) {
                return $.link(this, t, n, r, i, s, o, u)
            },
            unlink: function(t, n) {
                return $unlink(this, t, n)
            }
        }
    }
})();
