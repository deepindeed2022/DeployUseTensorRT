#ifndef DEPLOY_INCLUDE_DataBlob_H_
#define DEPLOY_INCLUDE_DataBlob_H_
#include <utility>
#include <memory>
#include <iostream>
class Size {
public:
	Size(size_t h, size_t w):
		m_height(h), m_width(w)
	{}
	Size():
		m_height(0), m_width(0)
	{}

	size_t height() const { return m_height; }
	size_t &height() { return m_height; }
	size_t width() const { return m_width; }
	size_t &width() { return m_width; }

	bool operator == (const Size &rhs) const {
		return height() == rhs.height() && width() == rhs.width();
	}
private:
	size_t m_height, m_width;
};

class DataBlobShape: public Size {
public:
	DataBlobShape(size_t heights, size_t widths, size_t channels):
		Size(heights, widths), m_channel(channels) { }
	DataBlobShape(size_t heights, size_t widths, size_t channels, size_t nums):
		Size(heights, widths), m_channel(channels) { }
	size_t channels() const { return m_channel; }
	bool operator == (const DataBlobShape &rhs) const {
		return Size::operator==(rhs) && channels() == rhs.channels();
	}
private:
	size_t m_channel;
};

template <typename T>
class DataBlob {
private:
	size_t m_height, m_width;
	size_t m_channel;
	size_t m_step;
	std::shared_ptr<T> m_data;
	size_t m_offset;
	T *raw_ptr() { return m_data.get() + m_offset; }
	const T *raw_ptr() const { return m_data.get() + m_offset; }
public:
	DataBlob();
	DataBlob(size_t height, size_t width, size_t channel, size_t step);
	DataBlob(size_t height, size_t width, size_t channel);
	// do not try to manage data by shared_ptr
	DataBlob(size_t height, size_t width, size_t channel, T *data);
	DataBlob(size_t height, size_t width, size_t channel, size_t step, T *data);
	// shallow-copy constructor
	DataBlob(const DataBlob<T> &rhs);
	DataBlob(const DataBlob<T> &rhs, size_t row_offset, size_t row_count,
			size_t col_offset, size_t col_count);
	DataBlob<T> &operator=(const DataBlob<T> &rhs);

	T &at(size_t r, size_t c, size_t ch);
	const T &at(size_t r, size_t c, size_t ch) const;

	DataBlob<T> clone() const;

	// read data from src
	void read(const T *src);
	// write data to dst
	void write(T *dst) const;

	const T *ptr(size_t r = 0) const {
		return raw_ptr() + r * m_step;
	}
	T *ptr(size_t r = 0) {
		return raw_ptr() + r * m_step;
	}
	size_t height() const { return m_height; }
	size_t width() const { return m_width;}
	size_t channels() const { return m_channel; }
	size_t step() const { return m_step; }
	size_t total_nr_elem() const {return height() * width() * channels();}
	size_t total_span_elem() const { return height() * step(); }
	bool equals(const DataBlob<T> &rhs) const;
	bool is_continuous() const;
	Size size() const { return {height(), width()}; }
	DataBlobShape shape() const { return {height(), width(), channels()}; }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const DataBlob<T> &m);

// type aliases
using uchar = unsigned char;
using ushort = unsigned short;
using DataBlob8u = DataBlob<uchar>;
using DataBlob32f = DataBlob<float>;

#endif