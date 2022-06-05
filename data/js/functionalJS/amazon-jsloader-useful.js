
(function(f,c){var b=[];function d(g){b.push(g)}function a(h){if(!h){return}var g=f.head||f.getElementsByTagName("head")[0]||f.documentElement,i=f.createElement("script");i.async="async";i.src=h;g.insertBefore(i,g.firstChild)}function e(){ue.uels=a;for(var g=0;g<b.length;g++){a(b[g])}ue.deffered=1}if(c.ue){ue.uels=d;if(c.ue.attach){c.ue.attach("load",e)}}})(document,window);
if (window.ue_csm) {
    window.ue_csm.useCel = 1;
    window.ue_csm.useCelFF = 20;
}

    if (window.P && P.load && P.load.js && P.when) {
    P.when('ready').execute(function() {
        P.load.js("http://z-ecx.images-amazon.com/images/G/01/browser-scripts/csm-all/csm-all-min-2447272645._V1_.js", ["csm_cel"]);
    });
}
 else if (window.ue && window.ue.uels) {
    ue.uels("http://z-ecx.images-amazon.com/images/G/01/browser-scripts/csm-all/csm-all-min-2447272645._V1_.js");
}
 else if (window.amznJQ) {
    amznJQ.addLogical('csm-all', ["http://z-ecx.images-amazon.com/images/G/01/browser-scripts/csm-all/csm-all-min-2447272645._V1_.js"]);
    amznJQ.available('csm-all', function() {});
}


var ue_tbno = 0,
ue_tble = 0,
ue_sstb = 0,
ue_ssle = 0,
ue_tbpv = 0;

(function(c,m){var b=("; expires="+new Date(+new Date()+604800000).toGMTString()),j,h=c.ue_sstb,p=c.ue_tbno,k=c.ue_tble,i=c.ue_tbpv,o=c.ue||{},e=i&&o.pageViz&&o.pageViz.event&&o.pageViz.propHid;function n(q){j=q;document.cookie="csm-hit="+q+("|"+(+new Date()))+b+"; path=/"}function l(){var r="",t=o.isBFT?"b":"s",u=""+o.oid,q=""+o.lid,s=u;if((u!=q)&&(q.length==20)){t+="a";s+=("-"+q)}if(h&&o.tabid){r=o.tabid+"+"}r+=(t+"-"+s);return r}function d(r){var q=l();if((p||(q!=j))&&(q.length<100)){n(q)}if(k){a(""+(r?r.type:"interaction")+" "+q)}}function g(){j=0;if(k){a("blur")}}function a(q){if(o.log){o.log(q,"csm")}}function f(q){if(m[o.pageViz.propHid]===false){g()}else{if(m[o.pageViz.propHid]===true){d({type:"visible"})}}}if(o.attach){o.attach("click",d);o.attach("keyup",d);if(!p){if(!e||(i==4)||(i==5)){o.attach("focus",d);o.attach("blur",g)}if(e){o.attach(o.pageViz.event,f)}if(e&&((i==3)||(i==5))){f({})}}}o.aftb=1})(ue_csm,document);
(function(c){var a="[CSM] Alert invocation detected with argument: ",e="WARN",b=c.alert;function d(){if(c.ueLogError){c.ueLogError({message:a+arguments[0],logLevel:e})}Function.prototype.apply.apply(b,[c,arguments||[]])}window.alert=d})(window);
(function(c,g,k){var n=c.ue,b=c.uex,h=0,j=0,l,m,e,f,a={click:1,mousemove:2,scroll:3,keydown:4};if(!n||!b){return}function p(r){if(j){return}j=a[r.type];if(typeof r.clientX==="undefined"){e=r.pageX;f=r.pageY}else{e=r.clientX;f=r.clientY}if(j==2&&(!l||(l==e&&m==f))){l=e;m=f;j=0;return}for(var q in a){if(a.hasOwnProperty(q)){n.detach(q,p,k)}}if(n.isl){g.setTimeout(function(){b("at",n.id)},0)}}function o(){var i="";if(!h&&j){h=1;i+="&ui="+j}return i}for(var d in a){if(a.hasOwnProperty(d)){n.attach(d,p,k)}}n._ui=o})(ue_csm,window,document);

