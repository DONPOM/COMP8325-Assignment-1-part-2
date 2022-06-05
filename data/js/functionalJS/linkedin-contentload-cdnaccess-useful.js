var ALLOWED_DOMAIN = null;
var ALTERNATE_SWF_LOCATION = null;

var domain = location.hostname;

// ensures a box is *.linkedin.biz, another possible domain
var whitelist = {
  '//{DOMAIN}/scds/common/u/lib/inject/0.4.2/relay.swf': [
    /^[a-z0-9\-_.]+\.licdn\.com$/i,
    /^[a-z0-9\-_.]+\.licdn-ei\.com$/i,
    /^[a-z0-9\-_.]+\.licdn-ei2\.com$/i,
    /^[a-z0-9\-_.]+\.linkedin\.biz$/i
  ],
  '//{DOMAIN}/u/lib/inject/0.4.2/relay.swf': [
    /^[a-z0-9\-_.]+\.lmodules\.com$/i,
    /^[a-z0-9\-_.]+\.lmodules-ei\.com$/i
  ],
  '//www.linkedin.com/scds/common/u/lib/inject/0.4.2/relay.swf': [
    /^www\.linkedin\.com$/i
  ],
  '//www.linkedin-ei.com/scds/common/u/lib/inject/0.4.2/relay.swf': [
    /^www\.linkedin-ei\.com$/i
  ],
  '//www.linkedin-ei2.com/scds/common/u/lib/inject/0.4.2/relay.swf': [
    /^www\.linkedin-ei2\.com$/i
  ],
  '//spdy.linkedin.com/scds/common/u/lib/inject/0.4.2/relay.swf': [
    /^spdy\.linkedin\.com$/i
  ]
  // unsupported: *.linkedin.com, *.linkedin-ei.com, *.slidesharecdn.com, *.slideshare.net
};

for (var path in whitelist) {
  if (ALTERNATE_SWF_LOCATION || !whitelist.hasOwnProperty(path)) {
    continue;
  }
  
  for (var i = 0, len = whitelist[path].length; i < len; i++) {
    if (whitelist[path][i].test(domain)) {
      ALTERNATE_SWF_LOCATION = path;
      break;
    }
  }
}

if (!ALTERNATE_SWF_LOCATION) {
  throw new Error('DOMAIN UNSUPPORTED: ' + domain);
}

ALTERNATE_SWF_LOCATION = (location.protocol) + ALTERNATE_SWF_LOCATION.replace(/\{DOMAIN\}/g, domain);
