
LJ.define('LJ.ATI');

xtnv   = document;
xtsd   = 'http://logc400';
xtsite = '528851';
xtn2   = '1';
xtpage = 'service::undef::undef::www.livejournal.com%2F';
xtdi   = '';

LJ.ATI.page = xtpage;
LJ.ATI.level2id = xtn2;

LJ.ATI.queryString = '';
LJ.ATI.params = {
    ac: '',
    an: '',

    x1:  '[' + encodeURIComponent(document.title) + ']',
    x2:  '[undef]',
    x3:  '2',
    x4:  '0',
    x5:  '[schemius]',
    x6:  '0',
    x7:  '[undef]',
    x8:  '0',
    x9:  '0',
    x10: '0',
    x11: '2',
    x12: '0',
    x13: '0',
    x14: '[undef]',
    x15: '0',
    x16: '2',
    x17: '[undef]',
    x18: '9',
    x19: '13',

    f1: '[noncyr]',
    f2: '2',
    f3: '2',
    f4: '2',

    tag: '[undef]',
    ptype: '1-0-0'
};
LJ.ATI.queryString = '&' + LiveJournal.constructUrl('', LJ.ATI.params).replace('?', '');

if (window.xtparam != null) {
    window.xtparam += LJ.ATI.queryString;
} else{
    window.xtparam  = LJ.ATI.queryString + '&ati=';
};

LJ.injectScript('http://l-stat.livejournal.net/js/ads/xtcore.js');
