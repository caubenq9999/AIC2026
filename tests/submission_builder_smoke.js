const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class FakeClassList {
    constructor() { this.values = new Set(); }
    add(...values) { values.forEach(value => this.values.add(value)); }
    remove(...values) { values.forEach(value => this.values.delete(value)); }
    contains(value) { return this.values.has(value); }
    toggle(value, force) {
        const enabled = force === undefined ? !this.contains(value) : Boolean(force);
        if (enabled) this.add(value); else this.remove(value);
        return enabled;
    }
}

class FakeElement {
    constructor(id = '') {
        this.id = id;
        this.value = '';
        this.files = [];
        this.children = [];
        this.classList = new FakeClassList();
        this.listeners = {};
        this.style = {};
        this.dataset = {};
        this.textContent = '';
        this.innerHTML = '';
    }
    addEventListener(name, callback) {
        (this.listeners[name] ||= []).push(callback);
    }
    async emit(name, overrides = {}) {
        for (const callback of this.listeners[name] || []) {
            await callback({ target: this, key: '', preventDefault() {}, ...overrides });
        }
    }
    appendChild(child) { this.children.push(child); return child; }
    append(...children) { this.children.push(...children); }
    prepend(...children) { this.children.unshift(...children); }
    querySelector() { return null; }
    querySelectorAll() { return []; }
    getBoundingClientRect() { return { top: 0, height: 40 }; }
    setAttribute() {}
    remove() {}
    click() {}
}

const root = path.resolve(__dirname, '..');
const elements = new Map();
const element = id => {
    if (!elements.has(id)) elements.set(id, new FakeElement(id));
    return elements.get(id);
};
const storage = new Map();
const windowListeners = {};
class TestURL extends URL {}
TestURL.createObjectURL = () => 'blob:test';
TestURL.revokeObjectURL = () => {};

const context = {
    console,
    Blob,
    URL: TestURL,
    setTimeout: callback => { callback(); return 1; },
    clearTimeout() {},
    confirm: () => true,
    fetch: async () => { throw new Error('Unexpected fetch in merge smoke test.'); },
    CustomEvent: class CustomEvent { constructor(type) { this.type = type; } },
    localStorage: {
        getItem: key => storage.get(key) ?? null,
        setItem: (key, value) => storage.set(key, String(value)),
    },
    document: {
        body: new FakeElement('body'),
        getElementById: element,
        createElement: () => new FakeElement(),
        addEventListener() {},
    },
    window: {
        location: { protocol: 'http:', origin: 'http://localhost:5000' },
        addEventListener(name, callback) { (windowListeners[name] ||= []).push(callback); },
        dispatchEvent(event) {
            for (const callback of windowListeners[event.type] || []) callback(event);
        },
        open: () => null,
    },
};
context.window.window = context.window;
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'submission-store.js'), 'utf8'), context);
context.PrelimSubmission = context.window.PrelimSubmission;

context.PrelimSubmission.save({
    version: 1,
    activeQueryId: 'query-1-kis',
    queries: {
        'query-1-kis': {
            id: 'query-1-kis', type: 'kis', eventCount: 0, prompt: 'local', answer: '',
            pinned: [{ videoId: 'L21_V001', frameIdx: 10, path: '' }],
            automatic: [], pool: [], neighborTimeBorder: 30,
        },
    },
    latestPools: { frame: [], trake: [], trakeEventCount: 0 },
});

vm.runInContext(fs.readFileSync(path.join(root, 'submission-builder.js'), 'utf8'), context);

async function run() {
    const projectInput = element('import-project');
    projectInput.files = [{
        text: async () => JSON.stringify({
            version: 1,
            activeQueryId: 'query-2-qa',
            queries: {
                'query-1-kis': {
                    id: 'query-1-kis', type: 'kis', pinned: [
                        { video_id: 'L21_V001', frame_idx: 10 },
                        { video_id: 'L21_V001', frame_idx: 20 },
                    ], automatic: [], pool: [],
                },
                'query-2-qa': {
                    id: 'query-2-qa', type: 'qa', answer: 'Hà Nội',
                    pinned: [{ video_id: 'L22_V003', frame_idx: 30, answer: 'Hà Nội' }],
                    automatic: [], pool: [],
                },
            },
            latestPools: {},
        }),
    }];
    await projectInput.emit('change');

    let state = context.PrelimSubmission.load();
    assert.deepEqual(
        Array.from(state.queries['query-1-kis'].pinned, item => item.frameIdx),
        [10, 20],
        'local pin must stay first and duplicate must be removed',
    );
    assert.equal(state.queries['query-2-qa'].pinned[0].videoId, 'L22_V003');
    assert.equal(state.queries['query-2-qa'].pinned[0].answer, 'Hà Nội');

    const pinnedRows = element('ranking-rows').children
        .filter(row => row.dataset.pinnedIndex !== undefined)
        .slice(-2);
    const transferValues = new Map();
    const dataTransfer = {
        setData: (key, value) => transferValues.set(key, value),
        getData: key => transferValues.get(key) || '',
        effectAllowed: '',
        dropEffect: '',
    };
    const firstDragHandle = pinnedRows[0].children[3].children[0];
    await firstDragHandle.emit('dragstart', { dataTransfer });
    await pinnedRows[1].emit('drop', { dataTransfer, clientY: 39 });
    state = context.PrelimSubmission.load();
    assert.deepEqual(
        Array.from(state.queries['query-1-kis'].pinned, item => item.frameIdx),
        [20, 10],
        'dragging below a row must reorder pinned candidates',
    );

    const beforeInvalid = JSON.stringify(state);
    projectInput.files = [{
        text: async () => JSON.stringify({
            version: 1,
            queries: {
                'broken-kis': { id: 'broken-kis', type: 'kis', pinned: [{ videoId: 'BAD' }] },
            },
        }),
    }];
    await projectInput.emit('change');
    assert.equal(JSON.stringify(context.PrelimSubmission.load()), beforeInvalid, 'invalid merge must be atomic');

    const folderInput = element('import-submission-folder');
    folderInput.files = [{
        name: 'query-3-trake.csv',
        webkitRelativePath: 'submission/query-3-trake.csv',
        text: async () => 'L23_V004,100,200,300\nL23_V005,110,210,310\n',
    }];
    await folderInput.emit('change');
    state = context.PrelimSubmission.load();
    assert.equal(state.queries['query-3-trake'].eventCount, 3);
    assert.deepEqual(
        Array.from(state.queries['query-3-trake'].pinned[0].frameIndices),
        [100, 200, 300],
    );
    console.log('submission merge smoke test OK');
}

run().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
