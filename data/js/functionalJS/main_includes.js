require([
    //base
    'foundation/consolefix',
    'foundation/framebuster',
    'foundation/hosts',

    //libraries
    'jquery/nyt',
    'underscore/nyt',
    'backbone/nyt',
    'hammer/nyt',
    'foundation/lib/sockjs',
    'foundation/cookies',

    //framework
    'foundation/views/page-manager',
    'foundation/views/base-view',
    'foundation/models/base-model',
    'foundation/collections/base-collection',

    //other foundation elements
    'foundation/models/user-data',
    'foundation/views/user-data',
    'foundation/models/page-storage',
    'foundation/views/websockets-transport',
    'foundation/lib/auth/userauth',
    'foundation/lib/polyfills/placeholder',
    'foundation/lib/polyfills/requestAnimationFrame'

]);

// needed to define this so this file isn't downloaded separately
define('foundation/main_includes',[],function() {});

