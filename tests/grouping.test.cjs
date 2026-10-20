// node --test tests/grouping.test.cjs
const test = require("node:test");
const assert = require("node:assert");
const { autoGroup } = require("../app/static/grouping.js");

const at = (s) => new Date(s).getTime();
const names = (groups) => groups.map((g) => g.map((f) => f.name));

test("OBS parts written back to back form one group", () => {
  const files = [
    { name: "2026-10-20 10-48-33.mkv", lastModified: at("2026-10-20T11:19:50") },
    { name: "2026-10-20 10-00-01.mkv", lastModified: at("2026-10-20T10:48:20") },
    { name: "2026-10-20 11-20-05.mkv", lastModified: at("2026-10-20T11:43:00") },
    { name: "2026-10-20 15-00-00.mkv", lastModified: at("2026-10-20T15:30:00") },
  ];
  assert.deepStrictEqual(names(autoGroup(files)), [
    ["2026-10-20 10-00-01.mkv", "2026-10-20 10-48-33.mkv", "2026-10-20 11-20-05.mkv"],
    ["2026-10-20 15-00-00.mkv"],
  ]);
});

test("explicit part markers group, plain numbering does not", () => {
  const files = ["call teil2.m4a", "call teil1.m4a", "notiz_01.opus", "notiz_02.opus", "x_part_3.mp3"]
    .map((name) => ({ name, lastModified: 0 }));
  assert.deepStrictEqual(names(autoGroup(files)), [
    ["call teil1.m4a", "call teil2.m4a"],
    ["notiz_01.opus"],
    ["notiz_02.opus"],
    ["x_part_3.mp3"],
  ]);
});
