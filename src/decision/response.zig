//! The decision response's wire format: answers into the JSON `nuclis decide
//! --json` writes, which the API serves byte for byte (timings aside). Each
//! answer's top level is exactly Jev's; extras sit under `nuclis`. Knows no
//! HTTP. docs/models/laya.md § nuclis decide.
const std = @import("std");
const inference = @import("inference");
const request_mod = @import("request.zig");

const profile = inference.profiles.laya;
const decide = inference.decide;

pub const schema_version = 1;

/// The model as the caller named it, and where its weights came from.
pub const Identity = struct { name: []const u8, repo: ?[]const u8 = null, revision: ?[]const u8 = null };

pub const Body = struct {
    request: request_mod.Request,
    results: []const decide.StateResult,
    identity: Identity,
    /// Opening the model for this call; zero when it was open already.
    load_ns: u64,
    timings: decide.Timings,
    explain: bool,
};

/// The `--json` body: one Jev response per state, then a newline.
pub fn write(out: *std.Io.Writer, body: Body) std.Io.Writer.Error!void {
    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try s.beginObject();
    try s.objectField("schema_version");
    try s.write(schema_version);
    try writeIdentity(&s, body.identity);
    try s.objectField("timings_ms");
    try s.beginObject();
    try s.objectField("load");
    try s.write(round1(ms(body.load_ns)));
    try s.objectField("tokenize");
    try s.write(round1(ms(body.timings.tokenize_ns)));
    try s.objectField("encode");
    try s.write(round1(ms(body.timings.encode_ns)));
    try s.endObject();
    try s.objectField("results");
    try s.beginArray();
    for (body.results, body.request.states) |result, labeled| {
        try s.beginObject();
        try writeAnswers(&s, body.request, result, if (body.explain) .explain else .nuclis);
        try writeUsage(&s, result);
        try s.objectField("nuclis");
        try s.beginObject();
        try s.objectField("state");
        try s.write(labeled.label);
        try s.objectField("state_tokens");
        try s.write(result.state_tokens);
        try s.objectField("truncated");
        try s.write(result.truncated);
        try s.endObject();
        try s.endObject();
    }
    try s.endArray();
    try s.endObject();
    try out.writeByte('\n');
}

/// Jev's single-state response, `{model, answers, usage}`, each answer with
/// Jev's fields only: the body of a client written against TypeSafe's
/// `systemone` call. Asserts one state.
pub fn writeSystemOne(out: *std.Io.Writer, body: Body) std.Io.Writer.Error!void {
    std.debug.assert(body.results.len == 1);
    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try s.beginObject();
    try s.objectField("model");
    try s.write(body.identity.name);
    try writeAnswers(&s, body.request, body.results[0], .jev);
    try writeUsage(&s, body.results[0]);
    try s.endObject();
    try out.writeByte('\n');
}

fn writeIdentity(s: *std.json.Stringify, identity: Identity) !void {
    try s.objectField("model");
    try s.write(identity.name);
    try s.objectField("repo");
    try s.write(identity.repo);
    try s.objectField("revision");
    try s.write(identity.revision);
}

/// What an answer carries beyond Jev's fields: nothing, the `nuclis`
/// object, or that object with the sequence's sizes.
const Detail = enum { jev, nuclis, explain };

fn writeAnswers(s: *std.json.Stringify, request: request_mod.Request, result: decide.StateResult, detail: Detail) !void {
    try s.objectField("answers");
    try s.beginObject();
    for (request.ids, request.questions, result.answers) |id, q, a| {
        try s.objectField(id);
        try writeAnswer(s, q, a, detail);
    }
    try s.endObject();
}

fn writeUsage(s: *std.json.Stringify, result: decide.StateResult) !void {
    try s.objectField("usage");
    try s.beginObject();
    try s.objectField("input_tokens");
    try s.write(result.input_tokens);
    try s.objectField("output_tokens");
    try s.write(0);
    try s.endObject();
}

fn writeAnswer(s: *std.json.Stringify, q: profile.Question, a: decide.Answer, detail: Detail) !void {
    const c = a.calibrated;
    try s.beginObject();
    try s.objectField("type");
    try s.write(@tagName(q.kind));
    switch (q.kind) {
        .choice => {
            try s.objectField("choice");
            try s.write(q.keys[c.best]);
        },
        .score => {
            try s.objectField("score");
            try s.write(profile.round4(c.value));
            try s.objectField("legend");
            try s.beginObject();
            for (q.keys, q.legend) |key, value| {
                try s.objectField(key);
                try s.write(value);
            }
            try s.endObject();
        },
        .noul => {
            try s.objectField("noul");
            try s.write(profile.round4(c.value));
        },
    }
    if (q.kind != .noul) {
        try s.objectField("probabilities");
        try s.beginObject();
        for (q.keys, c.probabilities) |key, p| {
            try s.objectField(key);
            try s.write(profile.round4(p));
        }
        try s.endObject();
    }
    // Jev's fields at the top (a noul has no confidence there); the
    // package's `answer_confidence` and the rest under `nuclis`.
    if (q.kind != .noul) {
        try s.objectField("confidence");
        try s.write(profile.round4(c.confidence));
    }
    if (detail == .jev) return s.endObject();
    try s.objectField("nuclis");
    try s.beginObject();
    try s.objectField("answer_confidence");
    try s.write(profile.round4(c.answer_confidence));
    try s.objectField("logits");
    try s.write(a.logits);
    try s.objectField("temperature");
    try s.write(c.temperature);
    try s.objectField("bucket");
    try s.write(a.bucket);
    if (detail == .explain) {
        if (a.sequence) |sequence| {
            try s.objectField("sequence_tokens");
            try s.write(sequence.ids.len);
            try s.objectField("state_kept");
            try s.write(sequence.state_kept);
        }
    }
    try s.endObject();
    try s.endObject();
}

pub fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

fn round1(x: f64) f64 {
    return @round(x * 10) / 10;
}
