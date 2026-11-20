/*! @rgrove/parse-xml v5.2.0 | ISC License | Copyright Ryan Grove */

// src/lib/StringScanner.ts
var emptyString = "";
var surrogatePair = /[\uD800-\uDBFF][\uDC00-\uDFFF]/g;
var StringScanner = class {
  constructor(string) {
    this.o = this.u(string, true);
    this.d = 0;
    this.length = string.length;
    this.p = this.o !== this.length;
    this.h = string;
    if (this.p) {
      let charsToBytes = [];
      for (let byteIndex = 0, charIndex = 0; charIndex <= this.o; ++charIndex) {
        charsToBytes[charIndex] = byteIndex;
        byteIndex += string.codePointAt(byteIndex) > 65535 ? 2 : 1;
      }
      this.C = charsToBytes;
    }
  }
  /**
   * Whether the current character index is at the end of the input string.
   */
  get D() {
    return this.d >= this.o;
  }
  // -- Protected Methods ------------------------------------------------------
  /**
   * Returns the number of characters in the given string, which may differ from
   * the byte length if the string contains multibyte characters.
   */
  u(string, multiByteSafe = this.p) {
    return multiByteSafe ? string.replace(surrogatePair, "_").length : string.length;
  }
  // -- Public Methods ---------------------------------------------------------
  /**
   * Advances the scanner by the given number of characters, stopping if the end
   * of the string is reached.
   */
  l(count = 1) {
    this.d = Math.min(this.o, this.d + count);
  }
  /**
   * Returns the byte index of the given character index in the string. The two
   * may differ in strings that contain multibyte characters.
   */
  f(charIndex = this.d) {
    var _a;
    return this.p ? (_a = this.C[charIndex]) != null ? _a : Infinity : charIndex;
  }
  /**
   * Consumes and returns the given number of characters if possible, advancing
   * the scanner and stopping if the end of the string is reached.
   *
   * If no characters could be consumed, an empty string will be returned.
   */
  G(charCount = 1) {
    let chars = this.i(charCount);
    this.l(charCount);
    return chars;
  }
  /**
   * Consumes and returns the given number of bytes if possible, advancing the
   * scanner and stopping if the end of the string is reached.
   *
   * It's up to the caller to ensure that the given byte count doesn't split a
   * multibyte character.
   *
   * If no bytes could be consumed, an empty string will be returned.
   */
  v(byteCount) {
    let byteIndex = this.f();
    let result = this.h.slice(byteIndex, byteIndex + byteCount);
    this.l(this.u(result));
    return result;
  }
  /**
   * Consumes and returns all characters for which the given function returns
   * `true`, stopping when `false` is returned or the end of the input is
   * reached.
   */
  w(fn) {
    let { length, p: multiByteMode, h: string } = this;
    let startByteIndex = this.f();
    let endByteIndex = startByteIndex;
    if (multiByteMode) {
      while (endByteIndex < length) {
        let char = string[endByteIndex];
        let isSurrogatePair = char >= "\uD800" && char <= "\uDBFF";
        if (isSurrogatePair) {
          char += string[endByteIndex + 1];
        }
        if (!fn(char)) {
          break;
        }
        endByteIndex += isSurrogatePair ? 2 : 1;
      }
    } else {
      while (endByteIndex < length && fn(string[endByteIndex])) {
        ++endByteIndex;
      }
    }
    return this.v(endByteIndex - startByteIndex);
  }
  /**
   * Consumes the given string if it exists at the current character index, and
   * advances the scanner.
   *
   * If the given string doesn't exist at the current character index, an empty
   * string will be returned and the scanner will not be advanced.
   */
  b(stringToConsume) {
    let { length } = stringToConsume;
    let byteIndex = this.f();
    if (stringToConsume === this.h.slice(byteIndex, byteIndex + length)) {
      this.l(length === 1 ? 1 : this.u(stringToConsume));
      return stringToConsume;
    }
    return emptyString;
  }
  /**
   * Consumes characters until the given global regex is matched, advancing the
   * scanner up to (but not beyond) the beginning of the match. If the regex
   * doesn't match, nothing will be consumed.
   *
   * Returns the consumed string, or an empty string if nothing was consumed.
   */
  x(regex) {
    let matchByteIndex = this.h.slice(this.f()).search(regex);
    return matchByteIndex > 0 ? this.v(matchByteIndex) : emptyString;
  }
  /**
   * Consumes characters until the given string is found, advancing the scanner
   * up to (but not beyond) that point. If the string is never found, nothing
   * will be consumed.
   *
   * Returns the consumed string, or an empty string if nothing was consumed.
   */
  s(searchString) {
    let byteIndex = this.f();
    let matchByteIndex = this.h.indexOf(searchString, byteIndex);
    return matchByteIndex > 0 ? this.v(matchByteIndex - byteIndex) : emptyString;
  }
  /**
   * Returns the given number of characters starting at the current character
   * index, without advancing the scanner and without exceeding the end of the
   * input string.
   */
  i(count = 1) {
    let { d: charIndex, h: string } = this;
    return this.p ? string.slice(this.f(charIndex), this.f(charIndex + count)) : string.slice(charIndex, charIndex + count);
  }
  /**
   * Resets the scanner position to the given character _index_, or to the start
   * of the input string if no index is given.
   *
   * If _index_ is negative, the scanner position will be moved backward by that
   * many characters, stopping if the beginning of the string is reached.
   */
  j(index = 0) {
    this.d = index >= 0 ? Math.min(this.o, index) : Math.max(0, this.d + index);
  }
};

// src/lib/syntax.ts
var attValueCharDoubleQuote = /["&<]/;
var attValueCharSingleQuote = /['&<]/;
var attValueNormalizedWhitespace = /\r\n|[\n\r\t]/g;
var endCharData = /<|&|]]>/;
var lineBreakToNormalize = /\r\n?/g;
var markupDeclarationDelimiter = /["'<>\]]|$/;
var predefinedEntities = Object.freeze(Object.assign(/* @__PURE__ */ Object.create(null), {
  amp: "&",
  apos: "'",
  gt: ">",
  lt: "<",
  quot: '"'
}));
function isNameChar(char) {
  let cp = char.codePointAt(0);
  return cp >= 97 && cp <= 122 || cp >= 65 && cp <= 90 || cp >= 48 && cp <= 57 || cp === 45 || cp === 46 || cp === 183 || cp >= 768 && cp <= 879 || cp === 8255 || cp === 8256 || isNameStartChar(char, cp);
}
function isNameStartChar(char, cp = char.codePointAt(0)) {
  return cp >= 97 && cp <= 122 || cp >= 65 && cp <= 90 || cp === 58 || cp === 95 || cp >= 192 && cp <= 214 || cp >= 216 && cp <= 246 || cp >= 248 && cp <= 767 || cp >= 880 && cp <= 893 || cp >= 895 && cp <= 8191 || cp === 8204 || cp === 8205 || cp >= 8304 && cp <= 8591 || cp >= 11264 && cp <= 12271 || cp >= 12289 && cp <= 55295 || cp >= 63744 && cp <= 64975 || cp >= 65008 && cp <= 65533 || cp >= 65536 && cp <= 983039;
}
function isReferenceChar(char) {
  return char === "#" || isNameChar(char);
}
function isWhitespace(char) {
  let cp = char.codePointAt(0);
  return cp === 32 || cp === 9 || cp === 10 || cp === 13;
}
function isXmlCodePoint(cp) {
  return cp >= 32 && cp <= 55295 || cp === 10 || cp === 9 || cp === 13 || cp >= 57344 && cp <= 65533 || cp >= 65536 && cp <= 1114111;
}

// src/lib/XmlNode.ts
var _XmlNode = class _XmlNode {
  constructor() {
    /**
     * Parent node of this node, or `null` if this node has no parent.
     */
    this.parent = null;
    /**
     * Starting byte offset of this node in the original XML string, or `-1` if
     * the offset is unknown.
     */
    this.start = -1;
    /**
     * Ending byte offset of this node in the original XML string, or `-1` if the
     * offset is unknown.
     */
    this.end = -1;
  }
  /**
   * Document that contains this node, or `null` if this node is not associated
   * with a document.
   */
  get document() {
    let node = this.parent;
    while (node !== null) {
      if (node.type === _XmlNode.TYPE_DOCUMENT) {
        return node;
      }
      node = node.parent;
    }
    return null;
  }
  /**
   * Whether this node is the root node of the document (also known as the
   * document element).
   */
  get isRootNode() {
    var _a;
    return ((_a = this.parent) == null ? void 0 : _a.type) === _XmlNode.TYPE_DOCUMENT && this.type === _XmlNode.TYPE_ELEMENT;
  }
  /**
   * Whether whitespace should be preserved in the content of this element and
   * its children.
   *
   * This is influenced by the value of the special `xml:space` attribute, and
   * will be `true` for any node whose `xml:space` attribute is set to
   * "preserve". If a node has no such attribute, it will inherit the value of
   * the nearest ancestor that does (if any).
   *
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#sec-white-space
   */
  get preserveWhitespace() {
    var _a;
    return !!((_a = this.parent) == null ? void 0 : _a.preserveWhitespace);
  }
  /**
   * Type of this node.
   *
   * The value of this property is a string that matches one of the static
   * `TYPE_*` properties on the `XmlNode` class (e.g. `TYPE_ELEMENT`,
   * `TYPE_TEXT`, etc.).
   *
   * The `XmlNode` class itself is a base class and doesn't have its own type
   * name.
   */
  get type() {
    return "";
  }
  /**
   * Returns a JSON-serializable object representing this node, minus properties
   * that could result in circular references.
   */
  toJSON() {
    let json = {
      type: this.type
    };
    if (this.isRootNode) {
      json.isRootNode = true;
    }
    if (this.preserveWhitespace) {
      json.preserveWhitespace = true;
    }
    if (this.start !== -1) {
      json.start = this.start;
      json.end = this.end;
    }
    return json;
  }
};
/**
 * Type value for an `XmlCdata` node.
 */
_XmlNode.TYPE_CDATA = "cdata";
/**
 * Type value for an `XmlComment` node.
 */
_XmlNode.TYPE_COMMENT = "comment";
/**
 * Type value for an `XmlDocument` node.
 */
_XmlNode.TYPE_DOCUMENT = "document";
/**
 * Type value for an `XmlDocumentType` node.
 */
_XmlNode.TYPE_DOCUMENT_TYPE = "doctype";
/**
 * Type value for an `XmlElement` node.
 */
_XmlNode.TYPE_ELEMENT = "element";
/**
 * Type value for an `XmlProcessingInstruction` node.
 */
_XmlNode.TYPE_PROCESSING_INSTRUCTION = "pi";
/**
 * Type value for an `XmlText` node.
 */
_XmlNode.TYPE_TEXT = "text";
/**
 * Type value for an `XmlDeclaration` node.
 */
_XmlNode.TYPE_XML_DECLARATION = "xmldecl";
var XmlNode = _XmlNode;

// src/lib/XmlText.ts
var XmlText = class extends XmlNode {
  constructor(text = "") {
    super();
    this.text = text;
  }
  get type() {
    return XmlNode.TYPE_TEXT;
  }
  toJSON() {
    return Object.assign(XmlNode.prototype.toJSON.call(this), {
      text: this.text
    });
  }
};

// src/lib/XmlCdata.ts
var XmlCdata = class extends XmlText {
  get type() {
    return XmlNode.TYPE_CDATA;
  }
};

// src/lib/XmlComment.ts
var XmlComment = class extends XmlNode {
  constructor(content = "") {
    super();
    this.content = content;
  }
  get type() {
    return XmlNode.TYPE_COMMENT;
  }
  toJSON() {
    return Object.assign(XmlNode.prototype.toJSON.call(this), {
      content: this.content
    });
  }
};

// src/lib/XmlDeclaration.ts
var XmlDeclaration = class extends XmlNode {
  constructor(version, encoding, standalone) {
    super();
    this.version = version;
    this.encoding = encoding != null ? encoding : null;
    this.standalone = standalone != null ? standalone : null;
  }
  get type() {
    return XmlNode.TYPE_XML_DECLARATION;
  }
  toJSON() {
    let json = XmlNode.prototype.toJSON.call(this);
    json.version = this.version;
    for (let key of ["encoding", "standalone"]) {
      if (this[key] !== null) {
        json[key] = this[key];
      }
    }
    return json;
  }
};

// src/lib/XmlElement.ts
var XmlElement = class _XmlElement extends XmlNode {
  constructor(name, attributes = /* @__PURE__ */ Object.create(null), children = []) {
    super();
    this.name = name;
    this.attributes = attributes;
    this.children = children;
  }
  /**
   * Whether this element is empty (meaning it has no children).
   */
  get isEmpty() {
    return this.children.length === 0;
  }
  get preserveWhitespace() {
    let node = this;
    while (node instanceof _XmlElement) {
      if ("xml:space" in node.attributes) {
        return node.attributes["xml:space"] === "preserve";
      }
      node = node.parent;
    }
    return false;
  }
  /**
   * Text content of this element and all its descendants.
   */
  get text() {
    return this.children.map((child) => "text" in child ? child.text : "").join("");
  }
  get type() {
    return XmlNode.TYPE_ELEMENT;
  }
  toJSON() {
    return Object.assign(XmlNode.prototype.toJSON.call(this), {
      name: this.name,
      attributes: this.attributes,
      children: this.children.map((child) => child.toJSON())
    });
  }
};

// src/lib/XmlDocument.ts
var XmlDocument = class extends XmlNode {
  constructor(children = []) {
    super();
    this.children = children;
  }
  get document() {
    return this;
  }
  /**
   * Root element of this document, or `null` if this document is empty.
   */
  get root() {
    for (let child of this.children) {
      if (child instanceof XmlElement) {
        return child;
      }
    }
    return null;
  }
  /**
   * Text content of this document and all its descendants.
   */
  get text() {
    return this.children.map((child) => "text" in child ? child.text : "").join("");
  }
  get type() {
    return XmlNode.TYPE_DOCUMENT;
  }
  toJSON() {
    return Object.assign(XmlNode.prototype.toJSON.call(this), {
      children: this.children.map((child) => child.toJSON())
    });
  }
};

// src/lib/XmlDocumentType.ts
var XmlDocumentType = class extends XmlNode {
  constructor(name, publicId, systemId, internalSubset) {
    super();
    this.name = name;
    this.publicId = publicId != null ? publicId : null;
    this.systemId = systemId != null ? systemId : null;
    this.internalSubset = internalSubset != null ? internalSubset : null;
  }
  get type() {
    return XmlNode.TYPE_DOCUMENT_TYPE;
  }
  toJSON() {
    let json = XmlNode.prototype.toJSON.call(this);
    json.name = this.name;
    for (let key of ["publicId", "systemId", "internalSubset"]) {
      if (this[key] !== null) {
        json[key] = this[key];
      }
    }
    return json;
  }
};

// src/lib/XmlError.ts
var XmlError = class extends Error {
  constructor(message, charIndex, xml) {
    let column = 1;
    let excerpt = "";
    let line = 1;
    for (let i = 0; i < charIndex; ++i) {
      let char = xml[i];
      if (char === "\n") {
        column = 1;
        excerpt = "";
        line += 1;
      } else {
        column += 1;
        excerpt += char;
      }
    }
    let eol = xml.indexOf("\n", charIndex);
    excerpt += eol === -1 ? xml.slice(charIndex) : xml.slice(charIndex, eol);
    let excerptStart = 0;
    if (excerpt.length > 50) {
      if (column < 40) {
        excerpt = excerpt.slice(0, 50);
      } else {
        excerptStart = column - 20;
        excerpt = excerpt.slice(excerptStart, column + 30);
      }
    }
    super(
      `${message} (line ${line}, column ${column})
  ${excerpt}
` + " ".repeat(column - excerptStart + 1) + "^\n"
    );
    this.column = column;
    this.excerpt = excerpt;
    this.line = line;
    this.name = "XmlError";
    this.pos = charIndex;
  }
};

// src/lib/XmlProcessingInstruction.ts
var XmlProcessingInstruction = class extends XmlNode {
  constructor(name, content = "") {
    super();
    this.name = name;
    this.content = content;
  }
  get type() {
    return XmlNode.TYPE_PROCESSING_INSTRUCTION;
  }
  toJSON() {
    return Object.assign(XmlNode.prototype.toJSON.call(this), {
      name: this.name,
      content: this.content
    });
  }
};

// src/lib/Parser.ts
var defaultMaxElementDepth = 256;
var emptyString2 = "";
var Parser = class {
  /**
   * @param xml XML string to parse.
   * @param options Parser options.
   */
  constructor(xml, options = {}) {
    this.depth = 0;
    var _a;
    let doc = this.document = new XmlDocument();
    this.m = doc;
    this.maxElementDepth = (_a = options.maxElementDepth) != null ? _a : defaultMaxElementDepth;
    this.g = options;
    this.c = new StringScanner(xml);
    if (this.g.includeOffsets) {
      doc.start = 0;
      doc.end = xml.length;
    }
    this.parse();
  }
  /**
   * Adds the given `XmlNode` as a child of `this.currentNode`.
   */
  k(node, charIndex) {
    node.parent = this.m;
    if (this.g.includeOffsets) {
      node.start = this.c.f(charIndex);
      node.end = this.c.f();
    }
    this.m.children.push(node);
    return true;
  }
  /**
   * Adds the given _text_ to the document, either by appending it to a
   * preceding `XmlText` node (if possible) or by creating a new `XmlText` node.
   *
   * When _normalize_ is `true` (the default), line breaks in _text_ are
   * normalized per section 2.11 of the XML spec. This must be `false` for text
   * that comes from a character or entity reference, since references aren't
   * subject to line break normalization.
   */
  y(text, charIndex, normalize = true) {
    let { children } = this.m;
    let { length } = children;
    if (normalize) {
      text = normalizeLineBreaks(text);
    }
    if (length > 0) {
      let prevNode = children[length - 1];
      if ((prevNode == null ? void 0 : prevNode.type) === XmlNode.TYPE_TEXT) {
        let textNode = prevNode;
        textNode.text += text;
        if (this.g.includeOffsets) {
          textNode.end = this.c.f();
        }
        return true;
      }
    }
    return this.k(new XmlText(text), charIndex);
  }
  /**
   * Consumes element attributes.
   *
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#sec-starttags
   */
  H() {
    let attributes = /* @__PURE__ */ Object.create(null);
    while (this.e()) {
      let attrName = this.q();
      if (!attrName) {
        break;
      }
      let attrValue = this.t() && this.I();
      if (attrValue === false) {
        throw this.a("Attribute value expected");
      }
      if (attrName in attributes) {
        throw this.a(`Duplicate attribute: ${attrName}`);
      }
      if (attrName === "xml:space" && attrValue !== "default" && attrValue !== "preserve") {
        throw this.a('Value of the `xml:space` attribute must be "default" or "preserve"');
      }
      attributes[attrName] = attrValue;
    }
    if (this.g.sortAttributes) {
      let attrNames = Object.keys(attributes).sort();
      let sortedAttributes = /* @__PURE__ */ Object.create(null);
      for (let i = 0; i < attrNames.length; ++i) {
        let attrName = attrNames[i];
        sortedAttributes[attrName] = attributes[attrName];
      }
      attributes = sortedAttributes;
    }
    return attributes;
  }
  /**
   * Consumes an `AttValue` (attribute value) if possible.
   *
   * @returns
   *   Contents of the `AttValue` minus quotes, or `false` if nothing was
   *   consumed. An empty string indicates that an `AttValue` was consumed but
   *   was empty.
   *
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-AttValue
   */
  I() {
    let { c: scanner } = this;
    let quote = scanner.i();
    if (quote !== '"' && quote !== "'") {
      return false;
    }
    scanner.l();
    let chars;
    let isClosed = false;
    let value = emptyString2;
    let regex = quote === '"' ? attValueCharDoubleQuote : attValueCharSingleQuote;
    matchLoop: while (!scanner.D) {
      chars = scanner.x(regex);
      if (chars) {
        this.n(chars);
        value += chars.replace(attValueNormalizedWhitespace, " ");
      }
      switch (scanner.i()) {
        case quote:
          isClosed = true;
          break matchLoop;
        case "&":
          value += this.E();
          continue;
        case "<":
          throw this.a("Unescaped `<` is not allowed in an attribute value");
        default:
          break matchLoop;
      }
    }
    if (!isClosed) {
      throw this.a("Unclosed attribute");
    }
    scanner.l();
    return value;
  }
  /**
   * Consumes a CDATA section if possible.
   *
   * @returns Whether a CDATA section was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#sec-cdata-sect
   */
  J() {
    let { c: scanner } = this;
    let startIndex = scanner.d;
    if (!scanner.b("<![CDATA[")) {
      return false;
    }
    let text = scanner.s("]]>");
    this.n(text);
    if (!scanner.b("]]>")) {
      throw this.a("Unclosed CDATA section");
    }
    return this.g.preserveCdata ? this.k(new XmlCdata(normalizeLineBreaks(text)), startIndex) : this.y(text, startIndex);
  }
  /**
   * Consumes character data if possible.
   *
   * @returns Whether character data was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#dt-chardata
   */
  K() {
    let { c: scanner } = this;
    let startIndex = scanner.d;
    let charData = scanner.x(endCharData);
    if (!charData) {
      return false;
    }
    this.n(charData);
    if (scanner.i(3) === "]]>") {
      throw this.a("Element content may not contain the CDATA section close delimiter `]]>`");
    }
    return this.y(charData, startIndex);
  }
  /**
   * Consumes a comment if possible.
   *
   * @param addToDocument
   *   Whether the comment may be added to the document (subject to the
   *   `preserveComments` option). This should be `false` for comments inside a
   *   DTD internal subset.
   *
   * @returns Whether a comment was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-Comment
   */
  z(addToDocument = true) {
    let { c: scanner } = this;
    let startIndex = scanner.d;
    if (!scanner.b("<!--")) {
      return false;
    }
    let content = scanner.s("--");
    this.n(content);
    if (!scanner.b("-->")) {
      if (scanner.i(2) === "--") {
        throw this.a("The string `--` isn't allowed inside a comment");
      }
      throw this.a("Unclosed comment");
    }
    return addToDocument && this.g.preserveComments ? this.k(new XmlComment(normalizeLineBreaks(content)), startIndex) : true;
  }
  /**
   * Consumes a reference in a content context if possible.
   *
   * This differs from `consumeReference()` in that a consumed reference will be
   * added to the document as a text node instead of returned.
   *
   * @returns Whether a reference was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#entproc
   */
  L() {
    let startIndex = this.c.d;
    let ref = this.E();
    return ref ? this.y(ref, startIndex, false) : false;
  }
  /**
   * Consumes a doctype declaration if possible.
   *
   * This is a loose implementation since DTDs are otherwise ignored. The
   * internal subset is only scanned closely enough to find where it ends.
   *
   * @returns Whether a doctype declaration was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#dtd
   */
  M() {
    let { c: scanner } = this;
    let startIndex = scanner.d;
    if (!scanner.b("<!DOCTYPE")) {
      return false;
    }
    let name = this.e() && this.q();
    if (!name) {
      throw this.a("Expected a name");
    }
    let publicId;
    let systemId;
    if (this.e()) {
      if (scanner.b("PUBLIC")) {
        publicId = this.e() && this.N();
        if (publicId === false) {
          throw this.a("Expected a public identifier");
        }
      }
      if (publicId !== void 0 || scanner.b("SYSTEM")) {
        systemId = this.e() && this.r();
        if (systemId === false) {
          throw this.a("Expected a system identifier");
        }
        this.e();
      }
    }
    let internalSubset;
    if (scanner.b("[")) {
      internalSubset = this.O();
      if (!scanner.b("]")) {
        throw this.a("Unclosed internal subset");
      }
      this.e();
    }
    if (!scanner.b(">")) {
      throw this.a("Unclosed doctype declaration");
    }
    return this.g.preserveDocumentType ? this.k(new XmlDocumentType(name, publicId, systemId, internalSubset), startIndex) : true;
  }
  /**
   * Consumes an element if possible.
   *
   * @returns Whether an element was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-element
   */
  F() {
    let { c: scanner } = this;
    let startIndex = scanner.d;
    if (!scanner.b("<")) {
      return false;
    }
    let name = this.q();
    if (!name) {
      scanner.j(startIndex);
      return false;
    }
    if (!(this.depth < this.maxElementDepth)) {
      scanner.j(startIndex);
      throw this.a(`Maximum element depth of ${this.maxElementDepth} exceeded (see the \`maxElementDepth\` option)`);
    }
    let attributes = this.H();
    let isEmpty = !!scanner.b("/>");
    let element = new XmlElement(name, attributes);
    element.parent = this.m;
    if (!isEmpty) {
      if (!scanner.b(">")) {
        throw this.a(`Unclosed start tag for element \`${name}\``);
      }
      this.m = element;
      this.depth += 1;
      do {
        this.K();
      } while (this.F() || this.L() || this.J() || this.A() || this.z());
      let endTagMark = scanner.d;
      let endTagName;
      if (!scanner.b("</") || !(endTagName = this.q()) || endTagName !== name) {
        scanner.j(endTagMark);
        throw this.a(`Missing end tag for element ${name}`);
      }
      this.e();
      if (!scanner.b(">")) {
        throw this.a(`Unclosed end tag for element ${name}`);
      }
      this.m = element.parent;
      this.depth -= 1;
    }
    return this.k(element, startIndex);
  }
  /**
   * Consumes an `Eq` production if possible.
   *
   * @returns Whether an `Eq` production was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-Eq
   */
  t() {
    this.e();
    if (this.c.b("=")) {
      this.e();
      return true;
    }
    return false;
  }
  /**
   * Consumes the contents of a DTD internal subset, stopping before the closing
   * `]` or at the first content that isn't allowed in an internal subset.
   *
   * The end of the internal subset is the first `]` that isn't inside a
   * comment, processing instruction, or quoted literal, all of which may
   * legally contain `]>`. Since DTDs are otherwise ignored, markup declarations
   * are only scanned closely enough to find where they end.
   *
   * @returns Raw text of the internal subset.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-intSubset
   */
  O() {
    let { c: scanner } = this;
    let startByteIndex = scanner.f();
    while (this.e() || this.z(false) || this.A(false) || this.P() || this.Q()) {
    }
    return scanner.h.slice(startByteIndex, scanner.f());
  }
  /**
   * Consumes an element, attribute-list, entity, or notation declaration in a
   * DTD internal subset if possible.
   *
   * The declaration isn't fully parsed. Quoted literals are skipped as opaque
   * units, and the declaration ends at the first `>` outside a literal.
   *
   * @returns Whether a markup declaration was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-markupdecl
   */
  P() {
    let { c: scanner } = this;
    let startIndex = scanner.d;
    if (scanner.b("<!ATTLIST")) {
      if (this.g.rejectAttributeListDeclarations) {
        scanner.j(startIndex);
        throw this.a("Attribute-list declarations aren't allowed when `rejectAttributeListDeclarations` is enabled");
      }
    } else if (!scanner.b("<!ELEMENT") && !scanner.b("<!ENTITY") && !scanner.b("<!NOTATION")) {
      return false;
    }
    for (; ; ) {
      this.n(scanner.x(markupDeclarationDelimiter));
      let char = scanner.i();
      if (char === ">") {
        scanner.l();
        return true;
      }
      if (char === '"' || char === "'") {
        this.r();
        continue;
      }
      throw this.a("Unclosed markup declaration");
    }
  }
  /**
   * Consumes `Misc` content if possible.
   *
   * @returns Whether anything was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-Misc
   */
  B() {
    return this.z() || this.A() || this.e();
  }
  /**
   * Consumes one or more `Name` characters if possible.
   *
   * @returns `Name` characters, or an empty string if none were consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-Name
   */
  q() {
    return isNameStartChar(this.c.i()) ? this.c.w(isNameChar) : emptyString2;
  }
  /**
   * Consumes a parameter entity reference in a DTD internal subset if possible.
   *
   * Since parse-xml doesn't expand parameter entities, a reference is rejected
   * when `rejectAttributeListDeclarations` is enabled, because its replacement
   * text could contain an attribute-list declaration.
   *
   * @returns Whether a parameter entity reference was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-PEReference
   */
  Q() {
    let { c: scanner } = this;
    if (scanner.i() !== "%") {
      return false;
    }
    if (this.g.rejectAttributeListDeclarations) {
      throw this.a("Parameter entity references aren't allowed when `rejectAttributeListDeclarations` is enabled");
    }
    scanner.l();
    if (!this.q() || !scanner.b(";")) {
      throw this.a("Invalid parameter entity reference");
    }
    return true;
  }
  /**
   * Consumes a processing instruction if possible.
   *
   * @param addToDocument
   *   Whether to add the processing instruction to the document. This should be
   *   `false` for processing instructions inside a DTD internal subset.
   *
   * @returns Whether a processing instruction was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#sec-pi
   */
  A(addToDocument = true) {
    let { c: scanner } = this;
    let startIndex = scanner.d;
    if (!scanner.b("<?")) {
      return false;
    }
    let name = this.q();
    if (name) {
      if (name.toLowerCase() === "xml") {
        scanner.j(startIndex);
        throw this.a("XML declaration isn't allowed here");
      }
    } else {
      throw this.a("Invalid processing instruction");
    }
    if (!this.e()) {
      if (scanner.b("?>")) {
        return addToDocument ? this.k(new XmlProcessingInstruction(name), startIndex) : true;
      }
      throw this.a("Whitespace is required after a processing instruction name");
    }
    let content = scanner.s("?>");
    this.n(content);
    if (!scanner.b("?>")) {
      throw this.a("Unterminated processing instruction");
    }
    return addToDocument ? this.k(new XmlProcessingInstruction(name, normalizeLineBreaks(content)), startIndex) : true;
  }
  /**
   * Consumes a prolog if possible.
   *
   * @returns Whether a prolog was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#sec-prolog-dtd
   */
  R() {
    let { c: scanner } = this;
    let startIndex = scanner.d;
    this.S();
    while (this.B()) {
    }
    if (this.M()) {
      while (this.B()) {
      }
    }
    return startIndex < scanner.d;
  }
  /**
   * Consumes a public identifier literal if possible.
   *
   * @returns
   *   Value of the public identifier literal minus quotes, or `false` if
   *   nothing was consumed. An empty string indicates that a public id literal
   *   was consumed but was empty.
   *
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-PubidLiteral
   */
  N() {
    let startIndex = this.c.d;
    let value = this.r();
    if (value !== false && !/^[-\x20\r\na-zA-Z0-9'()+,./:=?;!*#@$_%]*$/.test(value)) {
      this.c.j(startIndex);
      throw this.a("Invalid character in public identifier");
    }
    return value;
  }
  /**
   * Consumes a reference if possible.
   *
   * This differs from `consumeContentReference()` in that a consumed reference
   * will be returned rather than added to the document.
   *
   * @returns
   *   Parsed reference value, or `false` if nothing was consumed (to
   *   distinguish from a reference that resolves to an empty string).
   *
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-Reference
   */
  E() {
    let { c: scanner } = this;
    if (!scanner.b("&")) {
      return false;
    }
    let ref = scanner.w(isReferenceChar);
    if (scanner.G() !== ";") {
      throw this.a("Unterminated reference (a reference must end with `;`)");
    }
    let parsedValue;
    if (ref[0] === "#") {
      let codePoint = ref[1] === "x" ? parseInt(ref.slice(2), 16) : parseInt(ref.slice(1), 10);
      if (isNaN(codePoint) || !/^#(?:x[0-9A-Fa-f]+|[0-9]+)$/.test(ref)) {
        throw this.a("Invalid character reference");
      }
      if (!isXmlCodePoint(codePoint)) {
        throw this.a("Character reference resolves to an invalid character");
      }
      parsedValue = String.fromCodePoint(codePoint);
    } else {
      parsedValue = predefinedEntities[ref];
      if (parsedValue === void 0) {
        let {
          ignoreUndefinedEntities,
          resolveUndefinedEntity
        } = this.g;
        let wrappedRef = `&${ref};`;
        if (resolveUndefinedEntity) {
          let resolvedValue = resolveUndefinedEntity(wrappedRef);
          if (resolvedValue !== null && resolvedValue !== void 0) {
            let type = typeof resolvedValue;
            if (type !== "string") {
              throw new TypeError(`\`resolveUndefinedEntity()\` must return a string, \`null\`, or \`undefined\`, but returned a value of type ${type}`);
            }
            return resolvedValue;
          }
        }
        if (ignoreUndefinedEntities) {
          return wrappedRef;
        }
        scanner.j(-wrappedRef.length);
        throw this.a(`Named entity isn't defined: ${wrappedRef}`);
      }
    }
    return parsedValue;
  }
  /**
   * Consumes a `SystemLiteral` if possible.
   *
   * A `SystemLiteral` is similar to an attribute value, but allows the
   * characters `<` and `&` and doesn't replace references.
   *
   * @returns
   *   Value of the `SystemLiteral` minus quotes, or `false` if nothing was
   *   consumed. An empty string indicates that a `SystemLiteral` was consumed
   *   but was empty.
   *
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-SystemLiteral
   */
  r() {
    let { c: scanner } = this;
    let quote = scanner.b('"') || scanner.b("'");
    if (!quote) {
      return false;
    }
    let value = scanner.s(quote);
    this.n(value);
    if (!scanner.b(quote)) {
      throw this.a("Missing end quote");
    }
    return value;
  }
  /**
   * Consumes one or more whitespace characters if possible.
   *
   * @returns Whether any whitespace characters were consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#white
   */
  e() {
    return !!this.c.w(isWhitespace);
  }
  /**
   * Consumes an XML declaration if possible.
   *
   * @returns Whether an XML declaration was consumed.
   * @see https://www.w3.org/TR/2008/REC-xml-20081126/#NT-XMLDecl
   */
  S() {
    let { c: scanner } = this;
    let startIndex = scanner.d;
    if (!scanner.b("<?xml")) {
      return false;
    }
    if (isNameChar(scanner.i())) {
      scanner.j(startIndex);
      return false;
    }
    if (!this.e()) {
      throw this.a("Invalid XML declaration");
    }
    let version = !!scanner.b("version") && this.t() && this.r();
    if (version === false) {
      throw this.a("XML version is missing or invalid");
    } else if (!/^1\.[0-9]+$/.test(version)) {
      throw this.a("Invalid character in version number");
    }
    let encoding;
    let standalone;
    let hasWhitespace = this.e();
    if (hasWhitespace && scanner.b("encoding")) {
      let value = this.t() && this.r();
      if (value === false) {
        throw this.a("Invalid or unclosed XML declaration");
      } else if (!/^[A-Za-z][\w.-]*$/.test(value)) {
        throw this.a("Invalid character in encoding name");
      }
      encoding = value;
      hasWhitespace = this.e();
    }
    if (hasWhitespace && scanner.b("standalone")) {
      let value = this.t() && this.r();
      if (value === false) {
        throw this.a("Invalid or unclosed XML declaration");
      } else if (value !== "yes" && value !== "no") {
        throw this.a('Only "yes" and "no" are permitted as values of `standalone`');
      }
      standalone = value;
      this.e();
    }
    if (!scanner.b("?>")) {
      throw this.a("Invalid or unclosed XML declaration");
    }
    return this.g.preserveXmlDeclaration ? this.k(new XmlDeclaration(version, encoding, standalone), startIndex) : true;
  }
  /**
   * Returns an `XmlError` for the current scanner position.
   */
  a(message) {
    let { c: scanner } = this;
    return new XmlError(message, scanner.d, scanner.h);
  }
  /**
   * Parses the XML input.
   */
  parse() {
    this.c.b("\uFEFF");
    this.R();
    if (!this.F()) {
      throw this.a("Root element is missing or invalid");
    }
    while (this.B()) {
    }
    if (!this.c.D) {
      throw this.a("Extra content at the end of the document");
    }
  }
  /**
   * Throws an invalid character error if any character in the given _string_
   * isn't a valid XML character.
   */
  n(string) {
    let { length } = string;
    for (let i = 0; i < length; ++i) {
      let cp = string.codePointAt(i);
      if (!isXmlCodePoint(cp)) {
        this.c.j(-[...string.slice(i)].length);
        throw this.a("Invalid character");
      }
      if (cp > 65535) {
        i += 1;
      }
    }
  }
};
function normalizeLineBreaks(text) {
  return text.includes("\r") ? text.replace(lineBreakToNormalize, "\n") : text;
}

// src/index.ts
function parseXml(xml, options) {
  return new Parser(xml, options).document;
}
export {
  XmlCdata,
  XmlComment,
  XmlDeclaration,
  XmlDocument,
  XmlDocumentType,
  XmlElement,
  XmlError,
  XmlNode,
  XmlProcessingInstruction,
  XmlText,
  parseXml
};
