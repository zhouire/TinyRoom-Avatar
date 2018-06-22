#pragma once

#include "Win32_GLAppUtil.h";
#include "AvatarManager.h";
#include <vector>;
#include <map>;

struct Model
{
	struct Vertex
	{
		Vector3f  Pos;
		DWORD     C;
		float     U, V;
	};

	Vector3f        Pos;
	Quatf           Rot;
	Matrix4f        Mat;
	int             numVertices, numIndices;
	Vertex          Vertices[2000]; // Note fixed maximum
	GLushort        Indices[2000];
	ShaderFill    * Fill;
	VertexBuffer  * vertexBuffer;
	IndexBuffer   * indexBuffer;

	Model(Vector3f pos, ShaderFill * fill) :
		numVertices(0),
		numIndices(0),
		Pos(pos),
		Rot(),
		Mat(),
		Fill(fill),
		vertexBuffer(nullptr),
		indexBuffer(nullptr)
	{}

	~Model()
	{
		FreeBuffers();
	}

	Matrix4f& GetMatrix()
	{
		Mat = Matrix4f(Rot);
		Mat = Matrix4f::Translation(Pos) * Mat;
		return Mat;
	}

	void AddVertex(const Vertex& v) { Vertices[numVertices++] = v; }
	void AddIndex(GLushort a) { Indices[numIndices++] = a; }

	void AllocateBuffers()
	{
		vertexBuffer = new VertexBuffer(&Vertices[0], numVertices * sizeof(Vertices[0]));
		indexBuffer = new IndexBuffer(&Indices[0], numIndices * sizeof(Indices[0]));
	}

	void FreeBuffers()
	{
		delete vertexBuffer; vertexBuffer = nullptr;
		delete indexBuffer; indexBuffer = nullptr;
	}

	void AddSolidColorBox(float x1, float y1, float z1, float x2, float y2, float z2, DWORD c)
	{
		Vector3f Vert[][2] =
		{
			Vector3f(x1, y2, z1), Vector3f(z1, x1), Vector3f(x2, y2, z1), Vector3f(z1, x2),
			Vector3f(x2, y2, z2), Vector3f(z2, x2), Vector3f(x1, y2, z2), Vector3f(z2, x1),
			Vector3f(x1, y1, z1), Vector3f(z1, x1), Vector3f(x2, y1, z1), Vector3f(z1, x2),
			Vector3f(x2, y1, z2), Vector3f(z2, x2), Vector3f(x1, y1, z2), Vector3f(z2, x1),
			Vector3f(x1, y1, z2), Vector3f(z2, y1), Vector3f(x1, y1, z1), Vector3f(z1, y1),
			Vector3f(x1, y2, z1), Vector3f(z1, y2), Vector3f(x1, y2, z2), Vector3f(z2, y2),
			Vector3f(x2, y1, z2), Vector3f(z2, y1), Vector3f(x2, y1, z1), Vector3f(z1, y1),
			Vector3f(x2, y2, z1), Vector3f(z1, y2), Vector3f(x2, y2, z2), Vector3f(z2, y2),
			Vector3f(x1, y1, z1), Vector3f(x1, y1), Vector3f(x2, y1, z1), Vector3f(x2, y1),
			Vector3f(x2, y2, z1), Vector3f(x2, y2), Vector3f(x1, y2, z1), Vector3f(x1, y2),
			Vector3f(x1, y1, z2), Vector3f(x1, y1), Vector3f(x2, y1, z2), Vector3f(x2, y1),
			Vector3f(x2, y2, z2), Vector3f(x2, y2), Vector3f(x1, y2, z2), Vector3f(x1, y2)
		};

		GLushort CubeIndices[] =
		{
			0, 1, 3, 3, 1, 2,
			5, 4, 6, 6, 4, 7,
			8, 9, 11, 11, 9, 10,
			13, 12, 14, 14, 12, 15,
			16, 17, 19, 19, 17, 18,
			21, 20, 22, 22, 20, 23
		};

		for (int i = 0; i < sizeof(CubeIndices) / sizeof(CubeIndices[0]); ++i)
			AddIndex(CubeIndices[i] + GLushort(numVertices));

		// Generate a quad for each box face
		for (int v = 0; v < 6 * 4; v++)
		{
			// Make vertices, with some token lighting
			Vertex vvv; vvv.Pos = Vert[v][0]; vvv.U = Vert[v][1].x; vvv.V = Vert[v][1].y;
			float dist1 = (vvv.Pos - Vector3f(-2, 4, -2)).Length();
			float dist2 = (vvv.Pos - Vector3f(3, 4, -3)).Length();
			float dist3 = (vvv.Pos - Vector3f(-4, 3, 25)).Length();
			int   bri = rand() % 160;
			float B = ((c >> 16) & 0xff) * (bri + 192.0f * (0.65f + 8 / dist1 + 1 / dist2 + 4 / dist3)) / 255.0f;
			float G = ((c >> 8) & 0xff) * (bri + 192.0f * (0.65f + 8 / dist1 + 1 / dist2 + 4 / dist3)) / 255.0f;
			float R = ((c >> 0) & 0xff) * (bri + 192.0f * (0.65f + 8 / dist1 + 1 / dist2 + 4 / dist3)) / 255.0f;
			vvv.C = (c & 0xff000000) +
				((R > 255 ? 255 : DWORD(R)) << 16) +
				((G > 255 ? 255 : DWORD(G)) << 8) +
				(B > 255 ? 255 : DWORD(B));
			AddVertex(vvv);
		}
	}

	void Render(Matrix4f view, Matrix4f proj)
	{
		Matrix4f combined = proj * view * GetMatrix();

		glUseProgram(Fill->program);
		glUniform1i(glGetUniformLocation(Fill->program, "Texture0"), 0);
		glUniformMatrix4fv(glGetUniformLocation(Fill->program, "matWVP"), 1, GL_TRUE, (FLOAT*)&combined);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, Fill->texture->texId);

		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer->buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer->buffer);

		GLuint posLoc = glGetAttribLocation(Fill->program, "Position");
		GLuint colorLoc = glGetAttribLocation(Fill->program, "Color");
		GLuint uvLoc = glGetAttribLocation(Fill->program, "TexCoord");

		glEnableVertexAttribArray(posLoc);
		glEnableVertexAttribArray(colorLoc);
		glEnableVertexAttribArray(uvLoc);

		glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)OVR_OFFSETOF(Vertex, Pos));
		glVertexAttribPointer(colorLoc, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), (void*)OVR_OFFSETOF(Vertex, C));
		glVertexAttribPointer(uvLoc, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)OVR_OFFSETOF(Vertex, U));

		glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_SHORT, NULL);

		glDisableVertexAttribArray(posLoc);
		glDisableVertexAttribArray(colorLoc);
		glDisableVertexAttribArray(uvLoc);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		glUseProgram(0);
	}
};

//------------------------------------------------------------------------- 
struct Scene
{
	std::vector<Model*> Models;
	//use map for built-in find function
	std::map<Model*, int> removableModels;
	std::map<Model*, int> tempModels;
	
	float MARKER_SIZE = 0.2f;


	void    Add(Model * n)
	{
		Models.push_back(n);
	}

	void AddRemovable(Model * n)
	{
		//don't actually care about the values
		removableModels.insert(std::pair<Model*, int>(n, 1));
	}

	void AddTemp(Model * n)
	{
		tempModels.insert(std::pair<Model*, int>(n, 1));
	}

	void Render(Matrix4f view, Matrix4f proj)
	{
		for (int i = 0; i < Models.size(); ++i)
			Models[i]->Render(view, proj);

		//render removableModels as well
		for (auto const m : removableModels) {
			//auto model = *m.first;
			m.first->Render(view, proj);
		}

	}

	Model * CreateMarker(float size, DWORD color, Vector3f pos)
	{
		//ShaderFill * grid_material = CreateTextures();


		static const GLchar* VertexShaderSrc =
			"#version 150\n"
			"uniform mat4 matWVP;\n"
			"in      vec4 Position;\n"
			"in      vec4 Color;\n"
			"in      vec2 TexCoord;\n"
			"out     vec2 oTexCoord;\n"
			"out     vec4 oColor;\n"
			"void main()\n"
			"{\n"
			"   gl_Position = (matWVP * Position);\n"
			"   oTexCoord   = TexCoord;\n"
			"   oColor.rgb  = pow(Color.rgb, vec3(2.2));\n"   // convert from sRGB to linear
			"   oColor.a    = Color.a;\n"
			"}\n";

		static const char* FragmentShaderSrc =
			"#version 150\n"
			"uniform sampler2D Texture0;\n"
			"in      vec4      oColor;\n"
			"in      vec2      oTexCoord;\n"
			"out     vec4      FragColor;\n"
			"void main()\n"
			"{\n"
			"   FragColor = oColor * texture2D(Texture0, oTexCoord);\n"
			"}\n";

		GLuint    vshader = CreateShader(GL_VERTEX_SHADER, VertexShaderSrc);
		GLuint    fshader = CreateShader(GL_FRAGMENT_SHADER, FragmentShaderSrc);

		// Make textures
		ShaderFill * grid_material[4];
		for (int k = 0; k < 4; ++k)
		{
			static DWORD tex_pixels[256 * 256];
			for (int j = 0; j < 256; ++j)
			{
				for (int i = 0; i < 256; ++i)
				{
					if (k == 0) tex_pixels[j * 256 + i] = (((i >> 7) ^ (j >> 7)) & 1) ? 0xffb4b4b4 : 0xff505050;// floor
					if (k == 1) tex_pixels[j * 256 + i] = (((j / 4 & 15) == 0) || (((i / 4 & 15) == 0) && ((((i / 4 & 31) == 0) ^ ((j / 4 >> 4) & 1)) == 0)))
						? 0xff3c3c3c : 0xffb4b4b4;// wall
					if (k == 2) tex_pixels[j * 256 + i] = (i / 4 == 0 || j / 4 == 0) ? 0xff505050 : 0xffb4b4b4;// ceiling
					if (k == 3) tex_pixels[j * 256 + i] = 0xffffffff;// blank
				}
			}
			TextureBuffer * generated_texture = new TextureBuffer(false, Sizei(256, 256), 4, (unsigned char *)tex_pixels);
			grid_material[k] = new ShaderFill(vshader, fshader, generated_texture);
		}

		glDeleteShader(vshader);
		glDeleteShader(fshader);






		Model * marker = new Model(Vector3f(0, 0, 0), (grid_material[2]));
		marker->AddSolidColorBox(0, 0, 0, size, size, size, color);
		marker->AllocateBuffers();
		marker->Pos = pos;
		AddRemovable(marker);

		return marker;
	}

	void RemoveModel(Model * n)
	{
		removableModels.erase(n);
	}

	GLuint CreateShader(GLenum type, const GLchar* src)
	{
		GLuint shader = glCreateShader(type);

		glShaderSource(shader, 1, &src, NULL);
		glCompileShader(shader);

		GLint r;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &r);
		if (!r)
		{
			GLchar msg[1024];
			glGetShaderInfoLog(shader, sizeof(msg), 0, msg);
			if (msg[0]) {
				OVR_DEBUG_LOG(("Compiling shader failed: %s\n", msg));
			}
			return 0;
		}

		return shader;
	}

	ShaderFill * CreateTextures()
	{
		static const GLchar* VertexShaderSrc =
			"#version 150\n"
			"uniform mat4 matWVP;\n"
			"in      vec4 Position;\n"
			"in      vec4 Color;\n"
			"in      vec2 TexCoord;\n"
			"out     vec2 oTexCoord;\n"
			"out     vec4 oColor;\n"
			"void main()\n"
			"{\n"
			"   gl_Position = (matWVP * Position);\n"
			"   oTexCoord   = TexCoord;\n"
			"   oColor.rgb  = pow(Color.rgb, vec3(2.2));\n"   // convert from sRGB to linear
			"   oColor.a    = Color.a;\n"
			"}\n";

		static const char* FragmentShaderSrc =
			"#version 150\n"
			"uniform sampler2D Texture0;\n"
			"in      vec4      oColor;\n"
			"in      vec2      oTexCoord;\n"
			"out     vec4      FragColor;\n"
			"void main()\n"
			"{\n"
			"   FragColor = oColor * texture2D(Texture0, oTexCoord);\n"
			"}\n";

		GLuint    vshader = CreateShader(GL_VERTEX_SHADER, VertexShaderSrc);
		GLuint    fshader = CreateShader(GL_FRAGMENT_SHADER, FragmentShaderSrc);

		// Make textures
		ShaderFill * grid_material[4];
		for (int k = 0; k < 4; ++k)
		{
			static DWORD tex_pixels[256 * 256];
			for (int j = 0; j < 256; ++j)
			{
				for (int i = 0; i < 256; ++i)
				{
					if (k == 0) tex_pixels[j * 256 + i] = (((i >> 7) ^ (j >> 7)) & 1) ? 0xffb4b4b4 : 0xff505050;// floor
					if (k == 1) tex_pixels[j * 256 + i] = (((j / 4 & 15) == 0) || (((i / 4 & 15) == 0) && ((((i / 4 & 31) == 0) ^ ((j / 4 >> 4) & 1)) == 0)))
						? 0xff3c3c3c : 0xffb4b4b4;// wall
					if (k == 2) tex_pixels[j * 256 + i] = (i / 4 == 0 || j / 4 == 0) ? 0xff505050 : 0xffb4b4b4;// ceiling
					if (k == 3) tex_pixels[j * 256 + i] = 0xffffffff;// blank
				}
			}
			TextureBuffer * generated_texture = new TextureBuffer(false, Sizei(256, 256), 4, (unsigned char *)tex_pixels);
			grid_material[k] = new ShaderFill(vshader, fshader, generated_texture);
		}

		glDeleteShader(vshader);
		glDeleteShader(fshader);

		return *grid_material;
	}


	Model * ColorRemovableModel(Vector3f rightHandPos) 
	{
		for (auto const &m : removableModels) {
			auto model = *m.first;
			Vector3f modelPos = model.Pos;
			//might have to change this to be a target area, not so specific
			if (rightHandPos == modelPos) {
				RemoveModel(&model);
				//dark red: 	0xFF800000
				Model *newMarker = CreateMarker(MARKER_SIZE, 0xFF800000, modelPos);
				return newMarker;
			}

			else {
				return nullptr;
			}
		}
	}


	void ControllerActions(ovrSession session)
	{
		ovrInputState touchState;
		ovr_GetInputState(session, ovrControllerType_Active, &touchState);
		ovrTrackingState trackingState = ovr_GetTrackingState(session, 0.0, false);

		glm::vec3 hmdP = _glmFromOvrVector(trackingState.HeadPose.ThePose.Position);
		glm::quat hmdQ = _glmFromOvrQuat(trackingState.HeadPose.ThePose.Orientation);
		Vector3f ovr_leftP = trackingState.HandPoses[ovrHand_Left].ThePose.Position;
		glm::vec3 leftP = _glmFromOvrVector(ovr_leftP);
		glm::quat leftQ = _glmFromOvrQuat(trackingState.HandPoses[ovrHand_Left].ThePose.Orientation);
		Vector3f ovr_rightP = trackingState.HandPoses[ovrHand_Right].ThePose.Position;
		glm::vec3 rightP = _glmFromOvrVector(ovr_rightP);
		glm::quat rightQ = _glmFromOvrQuat(trackingState.HandPoses[ovrHand_Right].ThePose.Orientation);

		ovrAvatarTransform hmd;
		_ovrAvatarTransformFromGlm(hmdP, hmdQ, glm::vec3(1.0f), &hmd);

		ovrAvatarTransform left;
		_ovrAvatarTransformFromGlm(leftP, leftQ, glm::vec3(1.0f), &left);

		ovrAvatarTransform right;
		_ovrAvatarTransformFromGlm(rightP, rightQ, glm::vec3(1.0f), &right);

		ovrAvatarHandInputState inputStateLeft;
		_ovrAvatarHandInputStateFromOvr(left, touchState, ovrHand_Left, &inputStateLeft);

		ovrAvatarHandInputState inputStateRight;
		_ovrAvatarHandInputStateFromOvr(right, touchState, ovrHand_Right, &inputStateRight);

		if (inputStateRight.touchMask == ovrAvatarTouch_Pointing) {
			if (inputStateRight.buttonMask == ovrAvatarButton_One) {
				//pure green: 	0xff008000
				CreateMarker(MARKER_SIZE, 0xff008000, ovr_rightP);
			}
			if (inputStateRight.buttonMask == ovrAvatarButton_Two && ColorRemovableModel(ovr_rightP)) {
				//remove pointed-at marker
				RemoveModel(ColorRemovableModel(ovr_rightP));
			}
			//just pointing, not pressing A or B
			else {
				//TO DO: colorRemovableModel. If return false, create a new light green model. Store model in vector, remove all of these temp models at end of main loop
				if (!ColorRemovableModel(ovr_rightP)) {
					//light green:  0xFF00FF00
					Model *newMarker = CreateMarker(MARKER_SIZE, 0xFF00FF00, ovr_rightP);
					AddTemp(newMarker);
				}
			}
		}

		if (inputStateRight.buttonMask == ovrAvatarButton_One) {
			//pure green: 	0xff008000
			CreateMarker(MARKER_SIZE, 0xff008000, ovr_rightP);
		}

		/*
		if (inputStateRight.buttonMask == ovrAvatarButton_One) {
			Model *newMarker = CreateMarker(10.0f, 0xFF00FF00, Vector3f(0, 0, 0));
		}
		*/
	}

	//move all temp models to the current hand position
	void moveTempModels(ovrSession session) 
	{
		ovrInputState touchState;
		ovr_GetInputState(session, ovrControllerType_Active, &touchState);
		ovrTrackingState trackingState = ovr_GetTrackingState(session, 0.0, false);

		Vector3f ovr_rightP = trackingState.HandPoses[ovrHand_Right].ThePose.Position;

		for (auto const &m : removableModels) {
			auto model = m.first;
			model->Pos = ovr_rightP;
		}
	}


	
	void Init(int includeIntensiveGPUobject)
	{
		
		static const GLchar* VertexShaderSrc =
			"#version 150\n"
			"uniform mat4 matWVP;\n"
			"in      vec4 Position;\n"
			"in      vec4 Color;\n"
			"in      vec2 TexCoord;\n"
			"out     vec2 oTexCoord;\n"
			"out     vec4 oColor;\n"
			"void main()\n"
			"{\n"
			"   gl_Position = (matWVP * Position);\n"
			"   oTexCoord   = TexCoord;\n"
			"   oColor.rgb  = pow(Color.rgb, vec3(2.2));\n"   // convert from sRGB to linear
			"   oColor.a    = Color.a;\n"
			"}\n";

		static const char* FragmentShaderSrc =
			"#version 150\n"
			"uniform sampler2D Texture0;\n"
			"in      vec4      oColor;\n"
			"in      vec2      oTexCoord;\n"
			"out     vec4      FragColor;\n"
			"void main()\n"
			"{\n"
			"   FragColor = oColor * texture2D(Texture0, oTexCoord);\n"
			"}\n";

		GLuint    vshader = CreateShader(GL_VERTEX_SHADER, VertexShaderSrc);
		GLuint    fshader = CreateShader(GL_FRAGMENT_SHADER, FragmentShaderSrc);

		// Make textures
		ShaderFill * grid_material[4];
		for (int k = 0; k < 4; ++k)
		{
			static DWORD tex_pixels[256 * 256];
			for (int j = 0; j < 256; ++j)
			{
				for (int i = 0; i < 256; ++i)
				{
					if (k == 0) tex_pixels[j * 256 + i] = (((i >> 7) ^ (j >> 7)) & 1) ? 0xffb4b4b4 : 0xff505050;// floor
					if (k == 1) tex_pixels[j * 256 + i] = (((j / 4 & 15) == 0) || (((i / 4 & 15) == 0) && ((((i / 4 & 31) == 0) ^ ((j / 4 >> 4) & 1)) == 0)))
						? 0xff3c3c3c : 0xffb4b4b4;// wall
					if (k == 2) tex_pixels[j * 256 + i] = (i / 4 == 0 || j / 4 == 0) ? 0xff505050 : 0xffb4b4b4;// ceiling
					if (k == 3) tex_pixels[j * 256 + i] = 0xffffffff;// blank
				}
			}
			TextureBuffer * generated_texture = new TextureBuffer(false, Sizei(256, 256), 4, (unsigned char *)tex_pixels);
			grid_material[k] = new ShaderFill(vshader, fshader, generated_texture);
		}

		glDeleteShader(vshader);
		glDeleteShader(fshader);
		

		//ShaderFill * grid_material = CreateTextures();

		// Construct geometry
		Model * m = new Model(Vector3f(0, 0, 0), (grid_material[2]));  // Moving box
		m->AddSolidColorBox(0, 0, 0, +1.0f, +1.0f, 1.0f, 0xff404040);
		m->AllocateBuffers();
		Add(m);

		m = new Model(Vector3f(0, 0, 0), (grid_material[1]));  // Walls
		m->AddSolidColorBox(-10.1f, 0.0f, -20.0f, -10.0f, 4.0f, 20.0f, 0xff808080); // Left Wall
		m->AddSolidColorBox(-10.0f, -0.1f, -20.1f, 10.0f, 4.0f, -20.0f, 0xff808080); // Back Wall
		m->AddSolidColorBox(10.0f, -0.1f, -20.0f, 10.1f, 4.0f, 20.0f, 0xff808080); // Right Wall
		m->AllocateBuffers();
		Add(m);

		if (includeIntensiveGPUobject)
		{
			m = new Model(Vector3f(0, 0, 0), (grid_material[0]));  // Floors
			for (float depth = 0.0f; depth > -3.0f; depth -= 0.1f)
				m->AddSolidColorBox(9.0f, 0.5f, -depth, -9.0f, 3.5f, -depth, 0x10ff80ff); // Partition
			m->AllocateBuffers();
			Add(m);
		}

		m = new Model(Vector3f(0, 0, 0), (grid_material[0]));  // Floors
		m->AddSolidColorBox(-10.0f, -0.1f, -20.0f, 10.0f, 0.0f, 20.1f, 0xff808080); // Main floor
		m->AddSolidColorBox(-15.0f, -6.1f, 18.0f, 15.0f, -6.0f, 30.0f, 0xff808080); // Bottom floor
		m->AllocateBuffers();
		Add(m);

		m = new Model(Vector3f(0, 0, 0), (grid_material[2]));  // Ceiling
		m->AddSolidColorBox(-10.0f, 4.0f, -20.0f, 10.0f, 4.1f, 20.1f, 0xff808080);
		m->AllocateBuffers();
		Add(m);

		m = new Model(Vector3f(0, 0, 0), (grid_material[3]));  // Fixtures & furniture
		m->AddSolidColorBox(9.5f, 0.75f, 3.0f, 10.1f, 2.5f, 3.1f, 0xff383838);   // Right side shelf// Verticals
		m->AddSolidColorBox(9.5f, 0.95f, 3.7f, 10.1f, 2.75f, 3.8f, 0xff383838);   // Right side shelf
		m->AddSolidColorBox(9.55f, 1.20f, 2.5f, 10.1f, 1.30f, 3.75f, 0xff383838); // Right side shelf// Horizontals
		m->AddSolidColorBox(9.55f, 2.00f, 3.05f, 10.1f, 2.10f, 4.2f, 0xff383838); // Right side shelf
		m->AddSolidColorBox(5.0f, 1.1f, 20.0f, 10.0f, 1.2f, 20.1f, 0xff383838);   // Right railing   
		m->AddSolidColorBox(-10.0f, 1.1f, 20.0f, -5.0f, 1.2f, 20.1f, 0xff383838);   // Left railing  
		for (float f = 5.0f; f <= 9.0f; f += 1.0f)
		{
			m->AddSolidColorBox(f, 0.0f, 20.0f, f + 0.1f, 1.1f, 20.1f, 0xff505050);// Left Bars
			m->AddSolidColorBox(-f, 1.1f, 20.0f, -f - 0.1f, 0.0f, 20.1f, 0xff505050);// Right Bars
		}
		m->AddSolidColorBox(-1.8f, 0.8f, 1.0f, 0.0f, 0.7f, 0.0f, 0xff505000); // Table
		m->AddSolidColorBox(-1.8f, 0.0f, 0.0f, -1.7f, 0.7f, 0.1f, 0xff505000); // Table Leg 
		m->AddSolidColorBox(-1.8f, 0.7f, 1.0f, -1.7f, 0.0f, 0.9f, 0xff505000); // Table Leg 
		m->AddSolidColorBox(0.0f, 0.0f, 1.0f, -0.1f, 0.7f, 0.9f, 0xff505000); // Table Leg 
		m->AddSolidColorBox(0.0f, 0.7f, 0.0f, -0.1f, 0.0f, 0.1f, 0xff505000); // Table Leg 
		m->AddSolidColorBox(-1.4f, 0.5f, -1.1f, -0.8f, 0.55f, -0.5f, 0xff202050); // Chair Set
		m->AddSolidColorBox(-1.4f, 0.0f, -1.1f, -1.34f, 1.0f, -1.04f, 0xff202050); // Chair Leg 1
		m->AddSolidColorBox(-1.4f, 0.5f, -0.5f, -1.34f, 0.0f, -0.56f, 0xff202050); // Chair Leg 2
		m->AddSolidColorBox(-0.8f, 0.0f, -0.5f, -0.86f, 0.5f, -0.56f, 0xff202050); // Chair Leg 2
		m->AddSolidColorBox(-0.8f, 1.0f, -1.1f, -0.86f, 0.0f, -1.04f, 0xff202050); // Chair Leg 2
		m->AddSolidColorBox(-1.4f, 0.97f, -1.05f, -0.8f, 0.92f, -1.10f, 0xff202050); // Chair Back high bar

		for (float f = 3.0f; f <= 6.6f; f += 0.4f)
			m->AddSolidColorBox(-3, 0.0f, f, -2.9f, 1.3f, f + 0.1f, 0xff404040); // Posts

		m->AllocateBuffers();
		Add(m);
	}

	int numModels = Models.size();

	Scene() : numModels(0) {}
	Scene(bool includeIntensiveGPUobject) :
		numModels(0)
	{
		Init(includeIntensiveGPUobject);
	}
	void Release()
	{
		while (numModels-- > 0)
			delete Models[numModels];
	}
	~Scene()
	{
		Release();
	}
};